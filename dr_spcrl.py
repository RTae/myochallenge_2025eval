from __future__ import annotations

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.utils import explained_variance
from sb3_contrib import RecurrentPPO


# ============================================================
# Beta network (dual variable)
# ============================================================

class BetaNet(nn.Module):
    """beta_phi(s, a) >= 0 via softplus"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: th.Tensor, act: th.Tensor) -> th.Tensor:
        x = th.cat([obs, act], dim=1)
        return F.softplus(self.net(x)) + 1e-4


# ============================================================
# DR-SPCRL Recurrent PPO
# ============================================================

class DRSPCRLRecurrentPPO(RecurrentPPO):
    def __init__(
        self,
        *args,
        eps_start: float = 0.0,
        eps_budget: float = 1.0,
        lr_beta: float = 5e-4,
        beta_updates: int = 5,
        lr_curr: float = 1e-3,
        alpha: float = 0.1,
        **kwargs,
    ):
        self.epsilon = float(eps_start)
        self.eps_budget = float(eps_budget)
        self.lr_beta = float(lr_beta)
        self.beta_updates = int(beta_updates)
        self.lr_curr = float(lr_curr)
        self.alpha = float(alpha)
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        obs_dim = int(np.prod(self.observation_space.shape))
        act_dim = int(np.prod(self.action_space.shape))
        self.beta_net = BetaNet(obs_dim, act_dim).to(self.device)
        self.beta_opt = th.optim.Adam(self.beta_net.parameters(), lr=self.lr_beta)

    # ------------------------------------------------------------
    # Robust value
    # ------------------------------------------------------------
    def robust_value(self, values_1d: th.Tensor, beta_1d: th.Tensor) -> th.Tensor:
        """
        values_1d: (B,)
        beta_1d:   (B,)  (positive)
        Return:    (B,)
          V_rob = -β log(mean(exp(-V/β))) - β ε
        """
        values_1d = values_1d.view(-1)              # (B,)
        beta_1d = beta_1d.view(-1).clamp_min(1e-4)  # (B,)

        # (B,B): each row corresponds to a different beta_i
        exp_term = th.exp(-values_1d.unsqueeze(0) / beta_1d.unsqueeze(1))
        mean_exp = exp_term.mean(dim=1)  # (B,)

        v_rob = -beta_1d * th.log(mean_exp + 1e-12) - beta_1d * float(self.epsilon)
        return v_rob  # (B,)

    def train(self) -> None:
        self.policy.set_training_mode(True)

        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = (
            self.clip_range_vf(self._current_progress_remaining)
            if self.clip_range_vf is not None
            else None
        )

        # ========================================================
        # (A) Beta updates
        # ========================================================
        for _ in range(self.beta_updates):
            rollout_data = next(self.rollout_buffer.get(self.batch_size))

            mask = rollout_data.mask.view(-1) > 0.5
            if mask.sum().item() < 2:
                continue

            obs = rollout_data.observations[mask]
            act = rollout_data.actions[mask]

            # old_values often (B,1) -> make it (B,)
            with th.no_grad():
                values_old = rollout_data.old_values[mask].view(-1)

            beta = self.beta_net(obs, act).view(-1)  # (B,)
            v_rob = self.robust_value(values_old, beta)

            loss_beta = -v_rob.mean()

            self.beta_opt.zero_grad()
            loss_beta.backward()
            th.nn.utils.clip_grad_norm_(self.beta_net.parameters(), 1.0)
            self.beta_opt.step()

        # ========================================================
        # (B) PPO updates
        # ========================================================
        entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []

        for _epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                mask = rollout_data.mask.view(-1) > 0.5
                if mask.sum().item() < 2:
                    continue

                # Evaluate on full batch (recurrent)
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    rollout_data.actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )

                # Mask everything consistently
                obs = rollout_data.observations[mask]
                actions = rollout_data.actions[mask]

                # critical: squeeze to 1D
                values = values[mask].view(-1)                    # (B,)
                old_values = rollout_data.old_values[mask].view(-1)
                returns = rollout_data.returns[mask].view(-1)

                log_prob = log_prob[mask].view(-1)
                old_log_prob = rollout_data.old_log_prob[mask].view(-1)

                # Robust advantage/target
                with th.no_grad():
                    beta = self.beta_net(obs, actions).view(-1)   # (B,)
                    v_rob = self.robust_value(old_values, beta)   # (B,)
                    adv_rob = (v_rob - values).detach()
                    adv_rob = (adv_rob - adv_rob.mean()) / (adv_rob.std() + 1e-8)

                ratio = th.exp(log_prob - old_log_prob)
                pg_loss1 = adv_rob * ratio
                pg_loss2 = adv_rob * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                pg_loss = -th.mean(th.min(pg_loss1, pg_loss2))
                pg_losses.append(pg_loss.item())

                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # Value prediction (clip vf) — all 1D
                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = old_values + th.clamp(
                        values - old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )

                v_target = v_rob.detach().view(-1)  # (B,)
                value_loss = F.mse_loss(values_pred, v_target)
                value_losses.append(value_loss.item())

                # Entropy loss (mask correctly)
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)  # fallback
                else:
                    entropy_loss = -th.mean(entropy[mask].view(-1))
                entropy_losses.append(entropy_loss.item())

                loss = pg_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

        # ========================================================
        # (C) Epsilon curriculum update
        # ========================================================
        with th.no_grad():
            batch = next(self.rollout_buffer.get(self.batch_size))
            mask = batch.mask.view(-1) > 0.5
            if mask.sum().item() > 0:
                beta_bar = self.beta_net(batch.observations[mask], batch.actions[mask]).view(-1).mean().item()
            else:
                beta_bar = 0.0

        grad = beta_bar + 2.0 * self.alpha * (self.epsilon - self.eps_budget)
        self.epsilon = float(np.clip(self.epsilon - self.lr_curr * grad, 0.0, self.eps_budget))

        # logging
        self.logger.record("train/epsilon", self.epsilon)
        self.logger.record("train/beta_bar", beta_bar)
        self.logger.record("train/value_loss", float(np.mean(value_losses)) if value_losses else 0.0)
        self.logger.record("train/policy_loss", float(np.mean(pg_losses)) if pg_losses else 0.0)
        self.logger.record("train/entropy_loss", float(np.mean(entropy_losses)) if entropy_losses else 0.0)
        self.logger.record("train/clip_fraction", float(np.mean(clip_fractions)) if clip_fractions else 0.0)

        try:
            ev = explained_variance(
                self.rollout_buffer.returns.flatten(),
                self.rollout_buffer.values.flatten(),
            )
            self.logger.record("train/explained_variance", float(ev))
        except Exception:
            pass
