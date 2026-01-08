# This file is part of the original work by Mathis Group (2023).
# Source: [https://github.com/amathislab/lattice/tree/main]
#
# MIT License
#
# Copyright (c) 2023 Mathis Group
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import numpy as np
from typing import Optional
from torch import nn
from torch.distributions import MultivariateNormal
from typing import Tuple
from torch.distributions import Normal
from stable_baselines3.common.distributions import DiagGaussianDistribution, TanhBijector, StateDependentNoiseDistribution


class LatticeNoiseDistribution(DiagGaussianDistribution):
    """
    Like Lattice noise distribution, non-state-dependent. Does not allow time correlation, but
    it is more efficient.

    :param action_dim: Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        super().__init__(action_dim=action_dim)

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0, state_dependent: bool = False) -> Tuple[nn.Module, nn.Parameter]:
        self.mean_actions = nn.Linear(latent_dim, self.action_dim)
        self.std_init = torch.tensor(log_std_init).exp()
        if state_dependent:
            log_std = nn.Linear(latent_dim, self.action_dim + latent_dim)
        else:
            log_std = nn.Parameter(torch.zeros(self.action_dim + latent_dim), requires_grad=True)
        return self.mean_actions, log_std

    def proba_distribution(self, mean_actions: torch.Tensor, log_std: torch.Tensor) -> "LatticeNoiseDistribution":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        std = log_std.exp() * self.std_init
        action_variance = std[..., : self.action_dim] ** 2
        latent_variance = std[..., self.action_dim :] ** 2

        sigma_mat = (self.mean_actions.weight * latent_variance[..., None, :]).matmul(self.mean_actions.weight.T)
        sigma_mat[..., range(self.action_dim), range(self.action_dim)] += action_variance

        sigma_mat = 0.5 * (sigma_mat + sigma_mat.transpose(-1, -2))
        L, info = torch.linalg.cholesky_ex(sigma_mat)

        if info.any():
            # Level 2: Add Jitter 1e-6
            sigma_mat[..., range(self.action_dim), range(self.action_dim)] += 1e-6
            L, info = torch.linalg.cholesky_ex(sigma_mat)
            
            if info.any():
                # Level 3: Add Jitter 1e-4
                sigma_mat[..., range(self.action_dim), range(self.action_dim)] += 1e-4
                L, info = torch.linalg.cholesky_ex(sigma_mat)
                
                if info.any():
                    # Final Level: Total fallback to Diagonal
                    # We use a standard Normal but wrap it to look like a MVN
                    std_dev = (action_variance + 1e-3).sqrt()
                    self.distribution = torch.distributions.Independent(
                        torch.distributions.Normal(mean_actions, std_dev), 1
                    )
                    return self
                    
        self.distribution = MultivariateNormal(mean_actions, sigma_mat)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()


class SquashedLatticeNoiseDistribution(LatticeNoiseDistribution):
    """
    Lattice noise distribution, followed by a squashing function (tanh) to ensure bounds.

    :param action_dim: Dimension of the action space.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """
    def __init__(self, action_dim: int, epsilon: float = 1e-6):
        super().__init__(action_dim)
        self.epsilon = epsilon
        self.gaussian_actions: Optional[torch.Tensor] = None
        
    def log_prob(self, actions: torch.Tensor, gaussian_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Inverse tanh
        # Naive implementation (not stable): 0.5 * torch.log((1 + x) / (1 - x))
        # We use numpy to avoid numerical instability
        if gaussian_actions is None:
            # It will be clipped to avoid NaN when inversing tanh
            gaussian_actions = TanhBijector.inverse(actions)

        # Log likelihood for a Gaussian distribution
        log_prob = super().log_prob(gaussian_actions)
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= torch.sum(torch.log(1 - actions**2 + self.epsilon), dim=1)
        return log_prob
    
    def entropy(self) -> Optional[torch.Tensor]:
        # No analytical form,
        # entropy needs to be estimated using -log_prob.mean()
        return None

    def sample(self) -> torch.Tensor:
        # Reparametrization trick to pass gradients
        self.gaussian_actions = super().sample()
        return torch.tanh(self.gaussian_actions)

    def mode(self) -> torch.Tensor:
        self.gaussian_actions = super().mode()
        # Squash the output
        return torch.tanh(self.gaussian_actions)

    def log_prob_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(action, self.gaussian_actions)
        return action, log_prob

class LatticeStateDependentNoiseDistribution(StateDependentNoiseDistribution):
    """
    A safer Lattice MVN distribution:
      - robust PD covariance repair
      - clipping to avoid exploding covariance
      - optional debug checks
    """

    def __init__(
        self,
        action_dim: int,
        full_std: bool = False,
        use_expln: bool = True,
        squash_output: bool = False,
        learn_features: bool = False,
        epsilon: float = 1e-6,
        std_clip: Tuple[float, float] = (1e-3, 2.0),
        std_reg: float = 0.0,
        alpha: float = 0.5,
        latent_clip: float = 5.0,
        max_var: float = 2.0,
        jitter: float = 1e-5,
        max_mean_abs: Optional[float] = 10.0,  # clamp mean actions
        max_weight_abs: Optional[float] = 5.0,  # clamp W in covariance build
        debug: bool = False,
    ):
        super().__init__(
            action_dim=action_dim,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            epsilon=epsilon,
            learn_features=learn_features,
        )

        self.min_std, self.max_std = float(std_clip[0]), float(std_clip[1])
        self.std_reg = float(std_reg)
        self.alpha = float(alpha)

        self.latent_clip = float(latent_clip)
        self.max_var = float(max_var)
        self.jitter = float(jitter)

        self.max_mean_abs = None if max_mean_abs is None else float(max_mean_abs)
        self.max_weight_abs = None if max_weight_abs is None else float(max_weight_abs)

        self.debug = bool(debug)

        # SB3 fields set later
        self.mean_actions_net: Optional[nn.Module] = None
        self.clipped_mean_actions_net: Optional[nn.Module] = None
        self.latent_sde_dim: Optional[int] = None
        self._latent_sde: Optional[torch.Tensor] = None

        # sampled exploration weights
        self.corr_weights_dist: Optional[Normal] = None
        self.ind_weights_dist: Optional[Normal] = None
        self.corr_exploration_mat: Optional[torch.Tensor] = None
        self.ind_exploration_mat: Optional[torch.Tensor] = None
        self.corr_exploration_matrices: Optional[torch.Tensor] = None
        self.ind_exploration_matrices: Optional[torch.Tensor] = None

    # -------------------------
    # safety helpers
    # -------------------------
    def _safe_latent(self, latent: torch.Tensor) -> torch.Tensor:
        latent = torch.nan_to_num(latent, nan=0.0, posinf=0.0, neginf=0.0)
        if self.latent_clip and self.latent_clip > 0:
            latent = torch.clamp(latent, -self.latent_clip, self.latent_clip)
        return latent

    def _pd_repair(self, cov: torch.Tensor) -> torch.Tensor:
        """
        Make covariance PD by trying Cholesky with increasing jitter.
        If still failing, fallback to diagonal covariance.
        """
        cov = torch.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        cov = 0.5 * (cov + cov.transpose(-1, -2))  # symmetrize

        b, n, _ = cov.shape
        eye = torch.eye(n, device=cov.device, dtype=cov.dtype).unsqueeze(0)  # (1,n,n)

        jitter = self.jitter
        for _ in range(6):
            try:
                _ = torch.linalg.cholesky(cov + jitter * eye)
                return cov + jitter * eye
            except RuntimeError:
                jitter *= 10.0

        # fallback: diagonalize
        diag = torch.diagonal(cov, dim1=-2, dim2=-1)  # (b,n)
        diag = torch.nan_to_num(diag, nan=self.jitter, posinf=self.max_var, neginf=self.jitter)
        diag = torch.clamp(diag, min=self.jitter, max=self.max_var)
        return torch.diag_embed(diag) + self.jitter * eye

    # -------------------------
    # Std parametrization
    # -------------------------
    def get_std(self, log_std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          corr_std: (latent_sde_dim, latent_sde_dim)
          ind_std:  (latent_sde_dim, action_dim)
        """
        assert self.latent_sde_dim is not None, "Call proba_distribution_net() first."

        log_min = float(np.log(self.min_std))
        log_max = float(np.log(self.max_std))

        log_std = torch.nan_to_num(log_std, nan=log_min, posinf=log_max, neginf=log_min)
        log_std = torch.clamp(log_std, log_min, log_max)

        # correction from gSDE paper / SB3
        log_std = log_std - 0.5 * np.log(self.latent_sde_dim)

        if self.use_expln:
            below = torch.exp(log_std) * (log_std <= 0)
            safe = log_std * (log_std > 0) + self.epsilon
            above = (torch.log1p(safe) + 1.0) * (log_std > 0)
            std = below + above
        else:
            std = torch.exp(log_std)

        std = torch.nan_to_num(std, nan=self.min_std, posinf=self.max_std, neginf=self.min_std)
        std = std.clamp(min=self.min_std, max=self.max_std)

        if self.full_std:
            # std shape: (latent_sde_dim, latent_sde_dim + action_dim)
            assert std.shape == (self.latent_sde_dim, self.latent_sde_dim + self.action_dim), std.shape
            corr_std = std[:, : self.latent_sde_dim]
            ind_std = std[:, -self.action_dim :]
        else:
            # std shape: (latent_sde_dim, 2)
            assert std.shape == (self.latent_sde_dim, 2), std.shape
            corr_std = torch.ones(
                self.latent_sde_dim, self.latent_sde_dim, device=std.device, dtype=std.dtype
            ) * std[:, 0:1]
            ind_std = torch.ones(
                self.latent_sde_dim, self.action_dim, device=std.device, dtype=std.dtype
            ) * std[:, 1:]

        return corr_std, ind_std

    # -------------------------
    # Exploration matrix sampling
    # -------------------------
    def sample_weights(self, log_std: torch.Tensor, batch_size: int = 1) -> None:
        corr_std, ind_std = self.get_std(log_std)

        self.corr_weights_dist = Normal(torch.zeros_like(corr_std), corr_std)
        self.ind_weights_dist = Normal(torch.zeros_like(ind_std), ind_std)

        self.corr_exploration_mat = self.corr_weights_dist.rsample()
        self.ind_exploration_mat = self.ind_weights_dist.rsample()

        self.corr_exploration_matrices = self.corr_weights_dist.rsample((batch_size,))
        self.ind_exploration_matrices = self.ind_weights_dist.rsample((batch_size,))

    # -------------------------
    # Network creation (SB3 API)
    # -------------------------
    def proba_distribution_net(
        self,
        latent_dim: int,
        log_std_init: float = -0.5,
        latent_sde_dim: Optional[int] = None,
        clip_mean: float = 0.0,
    ) -> Tuple[nn.Module, nn.Parameter]:
        """
        Returns (mean_actions_net, log_std_param)
        """
        self.mean_actions_net = nn.Linear(latent_dim, self.action_dim)

        if clip_mean and clip_mean > 0:
            self.clipped_mean_actions_net = nn.Sequential(
                self.mean_actions_net,
                nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean),
            )
        else:
            self.clipped_mean_actions_net = self.mean_actions_net

        self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim

        log_std = (
            torch.ones(self.latent_sde_dim, self.latent_sde_dim + self.action_dim)
            if self.full_std
            else torch.ones(self.latent_sde_dim, 2)
        )
        log_std = nn.Parameter(log_std * float(log_std_init), requires_grad=True)

        # initial sample
        self.sample_weights(log_std, batch_size=1)
        return self.clipped_mean_actions_net, log_std

    # -------------------------
    # Distribution building (SB3 API)
    # -------------------------
    def proba_distribution(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        latent_sde: torch.Tensor,
    ) -> "LatticeStateDependentNoiseDistribution":
        assert self.mean_actions_net is not None, "Call proba_distribution_net() first."
        assert self.latent_sde_dim is not None, "Call proba_distribution_net() first."

        latent = latent_sde if self.learn_features else latent_sde.detach()
        latent = self._safe_latent(latent)
        self._latent_sde = latent

        corr_std, ind_std = self.get_std(log_std)

        # variances (safe)
        latent_sq = latent.pow(2)  # (batch, latent_dim)
        latent_corr_variance = torch.mm(latent_sq, corr_std.pow(2))  # (batch, latent_dim)
        latent_ind_variance = torch.mm(latent_sq, ind_std.pow(2)) + (self.std_reg ** 2)  # (batch, action_dim)

        # cap variances (important!)
        latent_corr_variance = torch.clamp(latent_corr_variance, 0.0, self.max_var)
        latent_ind_variance = torch.clamp(latent_ind_variance, 0.0, self.max_var)

        # covariance from correlated part
        W = self.mean_actions_net.weight  # (action_dim, latent_dim)
        if self.max_weight_abs is not None and self.max_weight_abs > 0:
            W = torch.clamp(W, -self.max_weight_abs, self.max_weight_abs)

        # (batch, action_dim, action_dim)
        sigma_mat = (self.alpha ** 2) * (W * latent_corr_variance[:, None, :]).matmul(W.T)

        # add independent variance to diagonal
        idx = torch.arange(self.action_dim, device=sigma_mat.device)
        sigma_mat[:, idx, idx] = sigma_mat[:, idx, idx] + latent_ind_variance

        sigma_mat = self._pd_repair(sigma_mat)

        mean_actions = torch.nan_to_num(mean_actions, nan=0.0, posinf=0.0, neginf=0.0)
        if self.max_mean_abs is not None and self.max_mean_abs > 0:
            mean_actions = torch.clamp(mean_actions, -self.max_mean_abs, self.max_mean_abs)

        if self.debug:
            if not torch.isfinite(sigma_mat).all():
                raise RuntimeError("sigma_mat has NaN/Inf")
            if not torch.isfinite(mean_actions).all():
                raise RuntimeError("mean_actions has NaN/Inf")
            _ = torch.linalg.cholesky(sigma_mat)

        self.distribution = MultivariateNormal(
            loc=mean_actions,
            covariance_matrix=sigma_mat,
            validate_args=False,
        )
        return self

    # -------------------------
    # Log prob / entropy (SB3 API)
    # -------------------------
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.bijector is not None:
            gaussian_actions = self.bijector.inverse(actions)
        else:
            gaussian_actions = actions

        gaussian_actions = torch.nan_to_num(gaussian_actions, nan=0.0, posinf=0.0, neginf=0.0)

        log_prob = self.distribution.log_prob(gaussian_actions)
        if self.bijector is not None:
            log_prob -= torch.sum(self.bijector.log_prob_correction(gaussian_actions), dim=1)
        return log_prob

    def entropy(self) -> torch.Tensor:
        if self.bijector is not None:
            return None
        return self.distribution.entropy()

    # -------------------------
    # Sampling (SB3 API)
    # -------------------------
    def get_noise(self, latent_sde, exploration_mat, exploration_matrices) -> torch.Tensor:
        latent = latent_sde if self.learn_features else latent_sde.detach()
        latent = self._safe_latent(latent)

        if len(latent) == 1 or len(latent) != len(exploration_matrices):
            return torch.mm(latent, exploration_mat)

        latent = latent.unsqueeze(dim=1)  # (batch, 1, n_features)
        noise = torch.bmm(latent, exploration_matrices)  # (batch, 1, n_actions)
        return noise.squeeze(dim=1)

    def sample(self) -> torch.Tensor:
        assert self._latent_sde is not None, "proba_distribution() must be called before sample()."
        latent_noise = self.alpha * self.get_noise(
            self._latent_sde, self.corr_exploration_mat, self.corr_exploration_matrices
        )
        action_noise = self.get_noise(
            self._latent_sde, self.ind_exploration_mat, self.ind_exploration_matrices
        )

        latent_input = self._safe_latent(self._latent_sde + latent_noise)
        actions = self.clipped_mean_actions_net(latent_input) + action_noise

        if self.bijector is not None:
            return self.bijector.forward(actions)
        return actions