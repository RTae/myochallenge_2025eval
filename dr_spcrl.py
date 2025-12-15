from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym

from stable_baselines3.common.utils import explained_variance

from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.type_aliases import RNNStates


# =========================
# Samples with next_obs
# =========================

@dataclass
class RecurrentRolloutBufferSamplesNext:
    observations: th.Tensor
    next_observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    episode_starts: th.Tensor
    lstm_states: RNNStates
    # mask for padded sequences (True=valid), exists in sb3-contrib buffers
    mask: th.Tensor


# =========================
# Beta network
# =========================

class BetaNet(nn.Module):
    """beta_phi(s,a) >= 0, implemented via softplus"""

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


# =========================
# Custom rollout buffer
# =========================

class DRSPCRLRecurrentRolloutBuffer(RecurrentRolloutBuffer):
    """
    Same as sb3-contrib RecurrentRolloutBuffer but stores next_observations too.
    """

    def reset(self) -> None:
        super().reset()
        self.next_observations = np.zeros_like(self.observations)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        lstm_states: RNNStates,
    ) -> None:
        # Store current obs the usual way
        super().add(obs, action, reward, episode_start, value, log_prob, lstm_states)
        # Store next obs at the same position
        self.next_observations[self.pos - 1] = np.array(next_obs).copy()

    def get(self, batch_size: Optional[int] = None) -> Generator[RecurrentRolloutBufferSamplesNext, None, None]:
        # This is mostly copied from sb3-contrib buffer.get(), but adds next_observations.
        assert self.full, "Rollout buffer must be full before sampling from it."

        # Prepare indices (same behavior as SB3)
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # Flatten (T, N, ...) -> (T*N, ...)
        obs = self.swap_and_flatten(self.observations)
        next_obs = self.swap_and_flatten(self.next_observations)
        actions = self.swap_and_flatten(self.actions)
        values = self.swap_and_flatten(self.values)
        log_probs = self.swap_and_flatten(self.log_probs)
        advantages = self.swap_and_flatten(self.advantages)
        returns = self.swap_and_flatten(self.returns)
        episode_starts = self.swap_and_flatten(self.episode_starts)
        masks = self.swap_and_flatten(self.mask)

        # LSTM states are kept as (n_steps, n_envs, n_layers, hidden) in buffer,
        # sb3-contrib provides helper to gather correct initial states per sequence.
        # We rely on the parent method to create sequence minibatches.
        start_idx = 0
        while start_idx < len(indices):
            batch_inds = indices[start_idx : start_idx + batch_size]

            # Create recurrent minibatch (keeps valid mask and initial lstm states)
            data = self._get_samples(batch_inds)

            yield RecurrentRolloutBufferSamplesNext(
                observations=data.observations,
                next_observations=self.to_torch(next_obs[batch_inds]),
                actions=data.actions,
                old_values=data.old_values,
                old_log_prob=data.old_log_prob,
                advantages=data.advantages,
                returns=data.returns,
                episode_starts=data.episode_starts,
                lstm_states=data.lstm_states,
                mask=data.mask,
            )

            start_idx += batch_size


# =========================
# DR-SPCRL RecurrentPPO
# =========================

class DRSPCRLRecurrentPPO(RecurrentPPO):
    """
    DR-SPCRL modification:
    - Adds beta_net(s,a)
    - Uses robust value backup for critic targets
    - Uses robust advantages
    - Updates epsilon by self-paced rule after each training iteration
    """

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

        # Replace rollout buffer with our next_obs version
        self.rollout_buffer = DRSPCRLRecurrentRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # Build beta_net
        obs_dim = int(np.prod(self.observation_space.shape))
        act_dim = int(np.prod(self.action_space.shape))
        self.beta_net = BetaNet(obs_dim, act_dim).to(self.device)
        self.beta_opt = th.optim.Adam(self.beta_net.parameters(), lr=self.lr_beta)

    @th.no_grad()
    def _predict_next_values(
        self,
        next_obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        """
        Predict V(next_obs) using the critic LSTM states.
        RecurrentPPO policies accept (obs, lstm_states, episode_starts).
        """
        return self.policy.predict_values(next_obs, lstm_states, episode_starts)

    def robust_value_backup(self, next_values: th.Tensor, beta: th.Tensor) -> th.Tensor:
        """
        Robust backup (KL ball, dual form) using minibatch approximation.

        next_values: (B,) or (B,1)
        beta: (B,1)

        We approximate:
            -beta * log(mean_j exp(-V(s'_j)/beta_i)) - beta * epsilon
        """
        next_values = next_values.view(-1)
        beta = beta.view(-1, 1)

        # Stable enough for typical value scales; if you get NaNs, clamp next_values.
        exp_term = th.exp(-next_values.unsqueeze(0) / beta)  # (B,B)
        mean_exp = exp_term.mean(dim=0)                      # (B,)
        v_rob = -beta.squeeze() * th.log(mean_exp + 1e-12) - beta.squeeze() * float(self.epsilon)
        return v_rob  # (B,)

    def collect_rollouts(self, env: gym.vector.VectorEnv, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        """
        Copy of RecurrentPPO.collect_rollouts but stores next_obs into buffer.
        We call super().collect_rollouts? Not possible because we need next_obs stored.
        """
        assert self._last_obs is not None
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs, self._last_lstm_states = self.policy.forward(
                    obs_tensor,
                    self._last_lstm_states,
                    self._last_episode_starts,
                )

            actions_np = actions.cpu().numpy()
            # Step env
            new_obs, rewards, dones, infos = env.step(actions_np)

            n_steps += 1

            # Handle timeouts (SB3 convention)
            for i, done in enumerate(dones):
                if done and infos[i].get("terminal_observation") is not None and infos[i].get("TimeLimit.truncated", False):
                    terminal_obs = infos[i]["terminal_observation"]
                    terminal_obs_tensor = th.as_tensor(terminal_obs).to(self.device).unsqueeze(0)
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(
                            terminal_obs_tensor, self._last_lstm_states, th.as_tensor([False]).to(self.device)
                        )
                    rewards[i] += self.gamma * terminal_value.cpu().numpy().squeeze()

            rollout_buffer.add(
                obs=self._last_obs,
                next_obs=new_obs,
                action=actions_np,
                reward=rewards,
                episode_start=self._last_episode_starts,
                value=values,
                log_prob=log_probs,
                lstm_states=self._last_lstm_states,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones

            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

        # Compute last values (for GAE bootstrapping in buffer)
        with th.no_grad():
            obs_tensor = th.as_tensor(self._last_obs).to(self.device)
            last_values = self.policy.predict_values(obs_tensor, self._last_lstm_states, self._last_episode_starts)

        rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=self._last_episode_starts)

        callback.on_rollout_end()
        return True

    def train(self) -> None:
        """
        Override RecurrentPPO.train() to:
        1) Update beta_net K times
        2) Use robust critic target and robust advantages
        3) Update epsilon curriculum
        """
        self.policy.set_training_mode(True)

        # Update LR schedules
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore

        # -------------------------
        # (A) Beta updates
        # -------------------------
        for _ in range(self.beta_updates):
            rollout_data = next(self.rollout_buffer.get(self.batch_size))

            obs = rollout_data.observations
            act = rollout_data.actions
            mask = rollout_data.mask

            # Only train on valid (non-padded) tokens
            valid = mask.view(-1) > 0.5
            if valid.sum().item() < 2:
                continue

            obs_v = obs[valid]
            act_v = act[valid]
            next_obs_v = rollout_data.next_observations[valid]

            with th.no_grad():
                next_values = self._predict_next_values(next_obs_v, rollout_data.lstm_states, rollout_data.episode_starts)[valid]

            beta = self.beta_net(obs_v, act_v)  # (Bv,1)
            v_rob = self.robust_value_backup(next_values, beta)  # (Bv,)

            loss_beta = -v_rob.mean()

            self.beta_opt.zero_grad()
            loss_beta.backward()
            th.nn.utils.clip_grad_norm_(self.beta_net.parameters(), 1.0)
            self.beta_opt.step()

        # -------------------------
        # (B) PPO updates
        # -------------------------
        entropy_losses = []
        pg_losses = []
        value_losses = []
        clip_fractions = []

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                obs = rollout_data.observations
                actions = rollout_data.actions
                old_log_prob = rollout_data.old_log_prob
                old_values = rollout_data.old_values
                mask = rollout_data.mask
                valid = mask.view(-1) > 0.5
                if valid.sum().item() < 2:
                    continue

                # Evaluate with recurrent policy
                values, log_prob, entropy = self.policy.evaluate_actions(
                    obs, actions, rollout_data.lstm_states, rollout_data.episode_starts
                )
                # Mask valid
                values = values[valid]
                log_prob = log_prob[valid]
                old_log_prob_v = old_log_prob[valid]
                old_values_v = old_values[valid]
                actions_v = actions[valid]
                obs_v = obs[valid]
                next_obs_v = rollout_data.next_observations[valid]

                # Robust target
                with th.no_grad():
                    next_values = self._predict_next_values(next_obs_v, rollout_data.lstm_states, rollout_data.episode_starts)[valid]
                    beta = self.beta_net(obs_v, actions_v).detach()
                    v_rob_next = self.robust_value_backup(next_values, beta)

                    # We need rewards and dones aligned; rollout_data does not expose rewards directly.
                    # So we use returns from buffer BUT replace advantages with robust ones by recomputing:
                    # robust_returns = (old_returns - old_adv + ???) is messy.
                    #
                    # Practical SB3-friendly approach:
                    # - Keep SB3 returns for stability
                    # - Replace advantages with robust_A = (robust_bootstrap - V)
                    #
                    # If you want exact paper targets, implement custom compute_returns_and_advantage
                    # in the buffer using v_rob_next and stored rewards/dones.

                # SB3 baseline return/adv (already computed)
                returns = rollout_data.returns[valid]

                # Replace advantages with robust advantage (simple, stable):
                # A_rob = returns_rob - V(s), where returns_rob ~ returns + (v_rob_next - next_values)
                # (cheap correction that injects robustness into the signal)
                with th.no_grad():
                    adv_rob = (returns + (v_rob_next - next_values).detach()) - values.detach()
                    adv_rob = (adv_rob - adv_rob.mean()) / (adv_rob.std() + 1e-8)

                # Ratio
                ratio = th.exp(log_prob - old_log_prob_v)
                # Clipped policy loss
                pg_loss1 = adv_rob * ratio
                pg_loss2 = adv_rob * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                pg_loss = -th.mean(th.min(pg_loss1, pg_loss2))
                pg_losses.append(pg_loss.item())

                # Clip fraction
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # Value loss (robust-ish target correction)
                # Target = returns + (v_rob_next - next_values)  (same correction)
                v_target = returns + (v_rob_next - next_values).detach()
                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = old_values_v + th.clamp(values - old_values_v, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(values_pred, v_target)
                value_losses.append(value_loss.item())

                # Entropy loss
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)  # fallback
                else:
                    entropy_loss = -th.mean(entropy[valid])
                entropy_losses.append(entropy_loss.item())

                loss = pg_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

        # -------------------------
        # (C) Epsilon curriculum update
        # -------------------------
        # Use a fresh minibatch estimate of beta_bar (valid tokens)
        with th.no_grad():
            batch = next(self.rollout_buffer.get(self.batch_size))
            valid = batch.mask.view(-1) > 0.5
            if valid.sum().item() >= 2:
                beta_bar = self.beta_net(batch.observations[valid], batch.actions[valid]).mean().item()
            else:
                beta_bar = 0.0

        grad = beta_bar + 2.0 * self.alpha * (self.epsilon - self.eps_budget)
        self.epsilon = float(np.clip(self.epsilon - self.lr_curr * grad, 0.0, self.eps_budget))

        # Log
        self.logger.record("train/epsilon", self.epsilon)
        self.logger.record("train/beta_bar", beta_bar)
        self.logger.record("train/policy_gradient_loss", float(np.mean(pg_losses)) if pg_losses else 0.0)
        self.logger.record("train/value_loss", float(np.mean(value_losses)) if value_losses else 0.0)
        self.logger.record("train/entropy_loss", float(np.mean(entropy_losses)) if entropy_losses else 0.0)
        self.logger.record("train/clip_fraction", float(np.mean(clip_fractions)) if clip_fractions else 0.0)

        # Approx explained variance using (SB3 returns, predicted values)
        # (still useful for debugging)
        try:
            ev = explained_variance(
                self.rollout_buffer.returns.flatten(),
                self.rollout_buffer.values.flatten(),
            )
            self.logger.record("train/explained_variance", float(ev))
        except Exception:
            pass


def main():
    # Replace this env with your MyoSuite env:
    # from myosuite.utils import gym as myo_gym
    # env = myo_gym.make("myoChallengeTableTennis-v0")
    env = gym.make("Pendulum-v1")  # placeholder; switch to MyoSuite Table Tennis

    # Recurrent policies need VecEnv in SB3; easiest:
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = DummyVecEnv([lambda: env])

    model = DRSPCRLRecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        verbose=1,
        n_steps=128,          # increase for MyoSuite (e.g. 1024)
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,

        # DR-SPCRL knobs
        eps_start=0.0,
        eps_budget=1.0,
        lr_beta=5e-4,
        beta_updates=5,
        lr_curr=1e-3,
        alpha=0.1,
    )

    model.learn(total_timesteps=200_000)


if __name__ == "__main__":
    main()
