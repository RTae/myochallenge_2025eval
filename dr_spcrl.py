from __future__ import annotations

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.utils import explained_variance
from sb3_contrib import RecurrentPPO


# ============================================================
# Beta Network
# ============================================================

class BetaNet(nn.Module):
    """Dual variable β(s,a) ≥ 0"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: th.Tensor, act: th.Tensor) -> th.Tensor:
        x = th.cat([obs, act], dim=1)
        return F.softplus(self.net(x)).squeeze(-1) + 1e-4


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
        target_beta: float = 1.0,
        **kwargs,
    ):
        self.epsilon = float(eps_start)
        self.eps_budget = float(eps_budget)
        self.lr_beta = lr_beta
        self.beta_updates = beta_updates
        self.lr_curr = lr_curr
        self.alpha = alpha
        self.target_beta = target_beta

        super().__init__(*args, **kwargs)

    # ---------------------------------------------------------
    # Setup
    # ---------------------------------------------------------

    def _setup_model(self) -> None:
        super()._setup_model()

        obs_dim = int(np.prod(self.observation_space.shape))
        act_dim = int(np.prod(self.action_space.shape))

        self.beta_net = BetaNet(obs_dim, act_dim).to(self.device)
        self.beta_opt = th.optim.Adam(self.beta_net.parameters(), lr=self.lr_beta)

    # ---------------------------------------------------------
    # Robust Bellman Backup (log-sum-exp)
    # ---------------------------------------------------------

    def robust_backup(self, next_values: th.Tensor, beta: th.Tensor) -> th.Tensor:
        """
        next_values: (B,)
        beta:        (B,)
        returns:     (B,)
        """
        next_values = next_values.view(-1)
        beta = beta.view(-1).clamp_min(1e-4)

        # log(mean exp(-V / β))
        x = -next_values.unsqueeze(0) / beta.unsqueeze(1)  # (B,B)
        lse = th.logsumexp(x, dim=1) - np.log(next_values.shape[0])

        return -beta * lse - beta * self.epsilon

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    def train(self) -> None:
        self.policy.set_training_mode(True)

        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = (
            self.clip_range_vf(self._current_progress_remaining)
            if self.clip_range_vf is not None
            else None
        )

        # ======================================================
        # (A) Train β network
        # ======================================================

        for _ in range(self.beta_updates):
            rollout = next(self.rollout_buffer.get(self.batch_size))
            mask = rollout.mask.view(-1) > 0.5
            if mask.sum() < 2:
                continue

            obs = rollout.observations[mask]
            act = rollout.actions[mask]

            with th.no_grad():
                next_vals = rollout.returns[mask].view(-1)

            beta = self.beta_net(obs, act)
            v_rob = self.robust_backup(next_vals, beta)

            loss_beta = -v_rob.mean()
            self.beta_opt.zero_grad()
            loss_beta.backward()
            th.nn.utils.clip_grad_norm_(self.beta_net.parameters(), 1.0)
            self.beta_opt.step()

        # ======================================================
        # (B) PPO updates
        # ======================================================

        pg_losses, value_losses, entropy_losses, clip_fracs = [], [], [], []

        for _ in range(self.n_epochs):
            for rollout in self.rollout_buffer.get(self.batch_size):

                mask = rollout.mask.view(-1) > 0.5
                if mask.sum() < 2:
                    continue

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout.observations,
                    rollout.actions,
                    rollout.lstm_states,
                    rollout.episode_starts,
                )

                obs = rollout.observations[mask]
                act = rollout.actions[mask]

                values = values[mask].view(-1)
                old_values = rollout.old_values[mask].view(-1)
                returns = rollout.returns[mask].view(-1)

                log_prob = log_prob[mask].view(-1)
                old_log_prob = rollout.old_log_prob[mask].view(-1)

                # ---------- Robust advantage ----------
                with th.no_grad():
                    beta = self.beta_net(obs, act)
                    v_rob = self.robust_backup(returns, beta)
                    adv = (v_rob - values)
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                ratio = th.exp(log_prob - old_log_prob)
                pg_loss = -th.mean(
                    th.min(
                        adv * ratio,
                        adv * th.clamp(ratio, 1 - clip_range, 1 + clip_range),
                    )
                )
                pg_losses.append(pg_loss.item())

                clip_fracs.append(
                    th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                )

                # ---------- Critic (standard PPO target) ----------
                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = old_values + th.clamp(
                        values - old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )

                value_loss = F.mse_loss(values_pred, returns)
                value_losses.append(value_loss.item())

                entropy_loss = -th.mean(entropy[mask]) if entropy is not None else 0.0
                entropy_losses.append(float(entropy_loss))

                loss = pg_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

        # ======================================================
        # (C) ε update (self-paced curriculum)
        # ======================================================

        with th.no_grad():
            batch = next(self.rollout_buffer.get(self.batch_size))
            mask = batch.mask.view(-1) > 0.5
            beta_bar = (
                self.beta_net(batch.observations[mask], batch.actions[mask]).mean().item()
                if mask.sum() > 0
                else 0.0
            )

        grad = (beta_bar - self.target_beta) + self.alpha * (self.epsilon - self.eps_budget)
        self.epsilon = float(np.clip(self.epsilon - self.lr_curr * grad, 0.0, self.eps_budget))

        # ======================================================
        # Logging
        # ======================================================

        self.logger.record("train/epsilon", self.epsilon)
        self.logger.record("train/beta_bar", beta_bar)
        self.logger.record("train/policy_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/clip_fraction", np.mean(clip_fracs))

        try:
            ev = explained_variance(
                self.rollout_buffer.returns.flatten(),
                self.rollout_buffer.values.flatten(),
            )
            self.logger.record("train/explained_variance", ev)
        except Exception:
            pass
