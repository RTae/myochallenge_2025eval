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

from stable_baselines3.common.preprocessing import get_action_dim
from lattice.distributions import (
    LatticeNoiseDistribution,
    LatticeStateDependentNoiseDistribution,
)
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

class LatticeActorCriticPolicy(RecurrentActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        use_lattice=True,
        std_clip=(1e-3, 10),
        expln_eps=1e-6,
        std_reg=0,
        alpha=1,
        **kwargs
    ):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        if use_lattice:
            if self.use_sde:
                self.dist_kwargs.update(
                    {
                        "epsilon": expln_eps,
                        "std_clip": std_clip,
                        "std_reg": std_reg,
                        "alpha": alpha,
                    }
                )
                self.action_dist = LatticeStateDependentNoiseDistribution(
                    get_action_dim(self.action_space), **self.dist_kwargs
                )
            else:
                self.action_dist = LatticeNoiseDistribution(
                    get_action_dim(self.action_space)
                )
            self._build(lr_schedule)
