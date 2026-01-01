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


class LatticeStateDependentNoiseDistribution(StateDependentNoiseDistribution):
    """
    Distribution class of Lattice exploration.
    Paper: Latent Exploration for Reinforcement Learning https://arxiv.org/abs/2305.20065
    
    [Fixed Version with NaN Guards for Stability]
    """
    def __init__(
        self,
        action_dim: int,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        learn_features: bool = False,
        epsilon: float = 1e-6,
        std_clip: Tuple[float, float] = (1e-3, 1.0),
        std_reg: float = 0.0,
        alpha: float = 1,
    ):
        super().__init__(
            action_dim=action_dim,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            epsilon=epsilon,
            learn_features=learn_features,
        )
        self.min_std, self.max_std = std_clip
        self.std_reg = std_reg
        self.alpha = alpha

    def get_std(self, log_std: torch.Tensor) -> torch.Tensor:
        """
        Get the standard deviation from the learned parameter.
        Includes safety checks for NaNs.
        """
        # --- FIX 1: Sanitize Input log_std ---
        if torch.isnan(log_std).any() or torch.isinf(log_std).any():
            # If parameters are broken, reset them to a safe default (-0.5 gives std ~0.6)
            log_std = torch.nan_to_num(log_std, nan=-0.5, posinf=2.0, neginf=-5.0)
        # -------------------------------------

        # Apply correction to remove scaling of action std as a function of the latent dimension
        log_std = log_std.clip(min=np.log(self.min_std), max=np.log(self.max_std))
        log_std = log_std - 0.5 * np.log(self.latent_sde_dim)

        if self.use_expln:
            below_threshold = torch.exp(log_std) * (log_std <= 0)
            safe_log_std = log_std * (log_std > 0) + self.epsilon
            above_threshold = (torch.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            std = torch.exp(log_std)

        # --- FIX 2: Sanitize Output std ---
        # Ensure positive and finite
        std = torch.nan_to_num(std, nan=1.0, posinf=10.0)
        std = torch.clamp(std, min=1e-6)
        # ----------------------------------

        if self.full_std:
            assert std.shape == (
                self.latent_sde_dim,
                self.latent_sde_dim + self.action_dim,
            )
            corr_std = std[:, : self.latent_sde_dim]
            ind_std = std[:, -self.action_dim :]
        else:
            assert std.shape == (self.latent_sde_dim, 2), std.shape
            corr_std = torch.ones(self.latent_sde_dim, self.latent_sde_dim).to(log_std.device) * std[:, 0:1]
            ind_std = torch.ones(self.latent_sde_dim, self.action_dim).to(log_std.device) * std[:, 1:]
            
        return corr_std, ind_std

    def sample_weights(self, log_std: torch.Tensor, batch_size: int = 1) -> None:
        """
        Sample weights for the noise exploration matrix.
        Includes safety checks for Normal distribution creation.
        """
        corr_std, ind_std = self.get_std(log_std)
        
        # --- FIX 3: Validated Normal Creation ---
        # Double check that we aren't passing garbage to Normal()
        # (get_std handles most of this, but redundancy saves crashes)
        if torch.isnan(corr_std).any() or (corr_std <= 0).any():
             corr_std = torch.nan_to_num(corr_std, nan=1.0).clamp(min=1e-6)
             
        if torch.isnan(ind_std).any() or (ind_std <= 0).any():
             ind_std = torch.nan_to_num(ind_std, nan=1.0).clamp(min=1e-6)
        # ----------------------------------------

        self.corr_weights_dist = Normal(torch.zeros_like(corr_std), corr_std)
        self.ind_weights_dist = Normal(torch.zeros_like(ind_std), ind_std)

        # Reparametrization trick to pass gradients
        self.corr_exploration_mat = self.corr_weights_dist.rsample()
        self.ind_exploration_mat = self.ind_weights_dist.rsample()

        # Pre-compute matrices in case of parallel exploration
        self.corr_exploration_matrices = self.corr_weights_dist.rsample((batch_size,))
        self.ind_exploration_matrices = self.ind_weights_dist.rsample((batch_size,))

    def proba_distribution_net(
        self,
        latent_dim: int,
        log_std_init: float = 0,
        latent_sde_dim: Optional[int] = None,
        clip_mean: float = 0,
    ) -> Tuple[nn.Module, nn.Parameter]:
        # Note: we always consider that the noise is based on the features of the last
        # layer, so latent_sde_dim is the same as latent_dim
        self.mean_actions_net = nn.Linear(latent_dim, self.action_dim)
        if clip_mean > 0:
            self.clipped_mean_actions_net = nn.Sequential(
                self.mean_actions_net,
                nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.clipped_mean_actions_net = self.mean_actions_net
        self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim

        log_std = (
            torch.ones(self.latent_sde_dim, self.latent_sde_dim + self.action_dim)
            if self.full_std
            else torch.ones(self.latent_sde_dim, 2)
        )

        # Transform it into a parameter so it can be optimized
        log_std = nn.Parameter(log_std * log_std_init, requires_grad=True)
        # Sample an exploration matrix
        self.sample_weights(log_std)
        return self.clipped_mean_actions_net, log_std

    def proba_distribution(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        latent_sde: torch.Tensor,
    ) -> "LatticeNoiseDistribution":
        
        # --- NAN GUARDS ---
        if torch.isnan(mean_actions).any():
            mean_actions = torch.nan_to_num(mean_actions, nan=0.0)
        if torch.isnan(latent_sde).any():
            latent_sde = torch.nan_to_num(latent_sde, nan=0.0)
        if torch.isnan(log_std).any():
             log_std = torch.nan_to_num(log_std, nan=-0.5)

        # Detach the last layer features
        self._latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        corr_std, ind_std = self.get_std(log_std)
        latent_corr_variance = torch.mm(self._latent_sde**2, corr_std**2)
        latent_ind_variance = torch.mm(self._latent_sde**2, ind_std**2) + self.std_reg**2

        # Correlated variance
        sigma_mat = self.alpha**2 * (self.mean_actions_net.weight * latent_corr_variance[:, None, :]).matmul(
            self.mean_actions_net.weight.T
        )
        # Independent variance
        sigma_mat[:, range(self.action_dim), range(self.action_dim)] += latent_ind_variance
        
        # --- STABILIZATION ---
        sigma_mat = 0.5 * (sigma_mat + sigma_mat.transpose(-1, -2))
        
        if torch.isnan(sigma_mat).any():
             sigma_mat = torch.eye(self.action_dim, device=sigma_mat.device).expand_as(sigma_mat) * 1e-3

        L, info = torch.linalg.cholesky_ex(sigma_mat)

        if info.any():
            sigma_mat[..., range(self.action_dim), range(self.action_dim)] += 1e-6
            L, info = torch.linalg.cholesky_ex(sigma_mat)
            
            if info.any():
                sigma_mat[..., range(self.action_dim), range(self.action_dim)] += 1e-4
                L, info = torch.linalg.cholesky_ex(sigma_mat)
                
                if info.any():
                    # Fallback to Diagonal
                    variance_diag = sigma_mat.diagonal(dim1=-2, dim2=-1)
                    std_dev = torch.clamp(variance_diag, min=1e-6).sqrt()
                    std_dev = torch.nan_to_num(std_dev, nan=1.0) # Safety
                    
                    self.distribution = torch.distributions.Independent(
                        torch.distributions.Normal(mean_actions, std_dev), 1
                    )
                    return self

        self.distribution = MultivariateNormal(loc=mean_actions, covariance_matrix=sigma_mat, validate_args=False)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.bijector is not None:
            gaussian_actions = self.bijector.inverse(actions)
        else:
            gaussian_actions = actions
        log_prob = self.distribution.log_prob(gaussian_actions)

        if self.bijector is not None:
            # Squash correction
            log_prob -= torch.sum(self.bijector.log_prob_correction(gaussian_actions), dim=1)
        return log_prob

    def entropy(self) -> torch.Tensor:
        if self.bijector is not None:
            return None
        return self.distribution.entropy()

    def get_noise(
        self,
        latent_sde: torch.Tensor,
        exploration_mat: torch.Tensor,
        exploration_matrices: torch.Tensor,
    ) -> torch.Tensor:
        latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        # Default case: only one exploration matrix
        if len(latent_sde) == 1 or len(latent_sde) != len(exploration_matrices):
            return torch.mm(latent_sde, exploration_mat)
        # Use batch matrix multiplication for efficient computation
        latent_sde = latent_sde.unsqueeze(dim=1)
        noise = torch.bmm(latent_sde, exploration_matrices)
        return noise.squeeze(dim=1)

    def sample(self) -> torch.Tensor:
        latent_noise = self.alpha * self.get_noise(self._latent_sde, self.corr_exploration_mat, self.corr_exploration_matrices)
        action_noise = self.get_noise(self._latent_sde, self.ind_exploration_mat, self.ind_exploration_matrices)
        actions = self.clipped_mean_actions_net(self._latent_sde + latent_noise) + action_noise
        if self.bijector is not None:
            return self.bijector.forward(actions)
        return actions