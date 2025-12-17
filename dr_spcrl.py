from __future__ import annotations

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.utils import explained_variance
from sb3_contrib import RecurrentPPO


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
        # cap to avoid runaway beta
        return th.clamp(F.softplus(self.net(x)), max=50.0) + 1e-4


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
        target_beta: float = 5.0,
        **kwargs,
    ):
        self.epsilon = float(eps_start)
        self.eps_budget = float(eps_budget)
        self.lr_beta = float(lr_beta)
        self.beta_updates = int(beta_updates)
        self.lr_curr = float(lr_curr)
        self.alpha = float(alpha)
        self.target_beta = float(target_beta)
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        obs_dim = int(np.prod(self.observation_space.shape))
        act_dim = int(np.prod(self.action_space.shape))
        self.beta_net = BetaNet(obs_dim, act_dim).to(self.device)
        self.beta_opt = th.optim.Adam(self.beta_net.parameters(), lr=self.lr_beta)

    # Robust operator: for a batch of values V_j, produce robustified value per i:
    # Vrob_i = -β_i log( mean_j exp(-V_j/β_i) ) - β_i ε
    def robustify(self, values_1d: th.Tensor, beta_1d: th.Tensor) -> th.Tensor:
        values_1d = values_1d.view(-1)                          # (B,)
        beta_1d = beta_1d.view(-1).clamp_min(1e-4)              # (B,)

        x = -values_1d.unsqueeze(0) / beta_1d.unsqueeze(1)      # (B,B)
        lse = th.logsumexp(x, dim=1) - np.log(values_1d.shape[0])  # (B,)

        v_rob = -beta_1d * lse - beta_1d * float(self.epsilon)  # (B,)
        return v_rob

    def train(self) -> None:
        self.policy.set_training_mode(True)

        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = (
            self.clip_range_vf(self._current_progress_remaining)
            if self.clip_range_vf is not None
            else None
        )

        # -------------------------
        # (A) Update beta network
        # -------------------------
        for _ in range(self.beta_updates):
            rollout = next(self.rollout_buffer.get(self.batch_size))
            mask = rollout.mask.view(-1) > 0.5
            if mask.sum().item() < 2:
                continue

            obs = rollout.observations[mask]
            act = rollout.actions[mask]

            # Use critic's old value predictions (not returns)
            with th.no_grad():
                v_old = rollout.old_values[mask].view(-1)  # (B,)

            beta = self.beta_net(obs, act).view(-1)        # (B,)
            v_rob = self.robustify(v_old, beta)            # (B,)
            loss_beta = -v_rob.mean()

            self.beta_opt.zero_grad()
            loss_beta.backward()
            th.nn.utils.clip_grad_norm_(self.beta_net.parameters(), 1.0)
            self.beta_opt.step()

        # -------------------------
        # (B) PPO updates
        # -------------------------
        pg_losses, value_losses, entropy_losses, clip_fracs = [], [], [], []

        for _epoch in range(self.n_epochs):
            for rollout in self.rollout_buffer.get(self.batch_size):
                mask = rollout.mask.view(-1) > 0.5
                if mask.sum().item() < 2:
                    continue

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout.observations,
                    rollout.actions,
                    rollout.lstm_states,
                    rollout.episode_starts,
                )

                # mask + flatten
                obs = rollout.observations[mask]
                act = rollout.actions[mask]

                values = values[mask].view(-1)                  # (B,)
                old_values = rollout.old_values[mask].view(-1)  # (B,)
                returns = rollout.returns[mask].view(-1)        # (B,)

                log_prob = log_prob[mask].view(-1)
                old_log_prob = rollout.old_log_prob[mask].view(-1)

                # robust advantage + robust value target
                with th.no_grad():
                    beta = self.beta_net(obs, act).view(-1)
                    v_rob_old = self.robustify(old_values, beta)      # (B,)
                    adv = (v_rob_old - values)
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                ratio = th.exp(log_prob - old_log_prob)
                pg_loss = -th.mean(
                    th.min(
                        adv * ratio,
                        adv * th.clamp(ratio, 1 - clip_range, 1 + clip_range),
                    )
                )
                pg_losses.append(pg_loss.item())
                clip_fracs.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())

                # critic: train towards ROBUST target (not plain returns)
                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = old_values + th.clamp(values - old_values, -clip_range_vf, clip_range_vf)

                v_target = v_rob_old.detach()  # robust target
                value_loss = F.mse_loss(values_pred, v_target)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy[mask].view(-1))
                entropy_losses.append(float(entropy_loss.item()))

                loss = pg_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

        # -------------------------
        # (C) epsilon update (stable)
        # -------------------------
        with th.no_grad():
            batch = next(self.rollout_buffer.get(self.batch_size))
            m = batch.mask.view(-1) > 0.5
            if m.sum().item() > 0:
                beta_bar = self.beta_net(batch.observations[m], batch.actions[m]).mean().item()
            else:
                beta_bar = 0.0

        # Increase epsilon when beta_bar is above target -> keeps beta bounded
        self.epsilon += self.lr_curr * (beta_bar - self.target_beta)
        # optional soft regularization towards eps_budget:
        self.epsilon -= self.lr_curr * self.alpha * (self.epsilon - self.eps_budget)

        self.epsilon = float(np.clip(self.epsilon, 0.0, self.eps_budget))

        # -------------------------
        # logging
        # -------------------------
        self.logger.record("train/epsilon", self.epsilon)
        self.logger.record("train/beta_bar", beta_bar)
        self.logger.record("train/policy_loss", float(np.mean(pg_losses)) if pg_losses else 0.0)
        self.logger.record("train/value_loss", float(np.mean(value_losses)) if value_losses else 0.0)
        self.logger.record("train/entropy_loss", float(np.mean(entropy_losses)) if entropy_losses else 0.0)
        self.logger.record("train/clip_fraction", float(np.mean(clip_fracs)) if clip_fracs else 0.0)

        try:
            ev = explained_variance(self.rollout_buffer.returns.flatten(), self.rollout_buffer.values.flatten())
            self.logger.record("train/explained_variance", float(ev))
        except Exception:
            pass
