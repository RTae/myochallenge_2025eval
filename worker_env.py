from random import random
from typing import Tuple, Dict, Optional

from myosuite.utils import gym as myo_gym
from loguru import logger
import numpy as np

from config import Config

class TableTennisWorker(myo_gym.Env):
    """
    Worker trained to reach ABSOLUTE targets.
    Uses proper curriculum based on success rate, not just episode count.
    """
    
    def __init__(self, config: Config, device: str = 'cpu'):
        self.config = config
        self.device = device
        
        # Create environment
        self.env = myo_gym.make(config.env_id)
        
        # Goal space (absolute targets)
        self.goal_low = np.array([0.0, -2.0, -1.0, 0.0, -5.0, -5.0, -5.0], dtype=np.float32)
        self.goal_high = np.array([3.0, 0.0, 1.0, 3.0, 5.0, 5.0, 5.0], dtype=np.float32)
        
        self.goal_space = myo_gym.spaces.Box(low=self.goal_low, high=self.goal_high, dtype=np.float32)
        
        # Observation space
        self.state_dim = 7  # paddle_pos(3) + paddle_vel(3) + time(1)
        self.observation_dim = self.state_dim + self.goal_space.shape[0]
        self.observation_space = myo_gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )
        
        # Action space
        self.action_space = self.env.action_space
        
        # Initialize
        self.current_goal = None
        self._initial_paddle_pos = None
        
        # Reward parameters
        self.position_weight = 2.0
        self.velocity_weight = 1.0
        self.timing_weight = 3.0
        self.success_bonus = 20.0
        
        self.training_stage = 0
        self.recent_successes = []  # Track success/failure of last 100 episodes
        self.total_episodes = 0     # Track total episodes for logging
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment and sample new target."""
        obs, info = self.env.reset(seed=seed)
        
        # Store initial state
        obs_dict = self.env.obs_dict
        self._initial_paddle_pos = obs_dict['paddle_pos'].copy()
        
        # Sample target based on current training stage
        self.current_goal = self._sample_absolute_target()
        
        # Increment episode counter
        self.total_episodes += 1
        
        # Prepare observation
        augmented_obs = self._augment_observation()
        info['goal'] = self.current_goal.copy()
        info['training_stage'] = self.training_stage
        info['total_episodes'] = self.total_episodes
        
        return augmented_obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take environment step with curriculum learning."""
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        # Calculate goal-conditioned reward
        goal_reward, goal_info = self._calculate_absolute_target_reward()
        goal_achieved = goal_info.get('goal_achieved', False)
        
        # Update success tracking
        self.recent_successes.append(1 if goal_achieved else 0)
        
        # Keep only last 100 episodes
        if len(self.recent_successes) > 100:
            self.recent_successes.pop(0)
        
        # Check for curriculum advancement
        if len(self.recent_successes) == 100:
            success_rate = np.mean(self.recent_successes)
            if success_rate > 0.6 and self.training_stage < 2:
                self.training_stage += 1
                logger.info(f"Advanced to training stage {self.training_stage} (success: {success_rate:.2f})")
                self.recent_successes = []  # Reset tracking for new stage
        
        # Combine rewards
        total_reward = goal_reward + 0.1 * base_reward
        
        # Prepare next observation
        augmented_obs = self._augment_observation()
        
        # Add info
        info.update({
            'goal_reward': float(goal_reward),
            'goal_achieved': goal_achieved,
            'training_stage': self.training_stage,
            'total_episodes': self.total_episodes,
            'recent_success_rate': np.mean(self.recent_successes) if self.recent_successes else 0.0,
            'position_error': goal_info.get('position_error', 0),
            'time_error': goal_info.get('time_error', 0)
        })
        
        return augmented_obs, float(total_reward), terminated, truncated, info
    
    def _augment_observation(self) -> np.ndarray:
        """Combine state and goal into single observation vector."""
        obs_dict = self.env.obs_dict
        
        # Extract current state
        paddle_pos = obs_dict['paddle_pos']  # shape (3,)
        paddle_vel = obs_dict['paddle_vel']  # shape (3,)
        current_time = obs_dict['time']      # scalar
        
        # Create state vector using hstack (handles scalars better)
        state = np.hstack([paddle_pos, paddle_vel, current_time])
        
        # Combine state with goal
        augmented_obs = np.hstack([state, self.current_goal])
        
        return augmented_obs.astype(np.float32)
    
    def _sample_absolute_target(self) -> np.ndarray:
        """
        Sample ABSOLUTE targets based on training stage.
        Training stage advances based on success rate, not just episode count.
        """
        obs_dict = self.env.obs_dict
        current_time = obs_dict['time']
        current_paddle_pos = obs_dict['paddle_pos']
        
        # Use training_stage (which is based on success rate)
        if self.training_stage == 0:
            # STAGE 0: Easy targets (until we achieve >60% success)
            target_time = current_time + np.random.uniform(0.1, 0.3)
            target_pos = current_paddle_pos + np.random.uniform(-0.2, 0.2, 3)
            target_vel = np.random.uniform(-1.0, 1.0, 3)
            
        elif self.training_stage == 1:
            # STAGE 1: Medium difficulty
            target_time = current_time + np.random.uniform(0.15, 0.4)
            
            table_positions = [
                np.array([-1.0, -0.5, 1.2], dtype=np.float32),   # Back left
                np.array([-1.0, 0.0, 1.2], dtype=np.float32),    # Back center
                np.array([-1.0, 0.5, 1.2], dtype=np.float32),    # Back right
                np.array([-0.5, -0.5, 1.5], dtype=np.float32),   # Front left
                np.array([-0.5, 0.0, 1.5], dtype=np.float32),    # Front center
                np.array([-0.5, 0.5, 1.5], dtype=np.float32)     # Front right
            ]
            target_pos = random.choice(table_positions) + np.random.uniform(-0.1, 0.1, 3)
            target_vel = np.random.uniform(-2.0, 2.0, 3)
            
        else:
            # STAGE 2: Full difficulty
            target_time = current_time + np.random.uniform(0.1, 0.5)
            
            target_pos = np.array([
                np.random.uniform(-1.5, -0.3),   # x: back to near net
                np.random.uniform(-0.8, 0.8),    # y: left to right
                np.random.uniform(0.8, 2.0)      # z: low to high
            ], dtype=np.float32)
            
            target_vel = np.random.uniform(-3.0, 3.0, 3)
        
        # Create goal array
        goal = np.array([
            float(target_time),
            float(target_pos[0]),
            float(target_pos[1]), 
            float(target_pos[2]),
            float(target_vel[0]),
            float(target_vel[1]), 
            float(target_vel[2])
        ], dtype=np.float32)
        
        return np.clip(goal, self.goal_low, self.goal_high)
    
    def _calculate_absolute_target_reward(self) -> Tuple[float, Dict]:
        """
        Reward based on reaching absolute target by specified time.
        Uses obs_dict['time'] for current episode time.
        """
        obs_dict = self.env.obs_dict
        
        # Current state
        paddle_pos = obs_dict['paddle_pos']
        paddle_vel = obs_dict['paddle_vel']
        current_time = obs_dict['time']
        
        # Parse goal
        target_time = self.current_goal[0]
        target_pos = self.current_goal[1:4]
        target_vel = self.current_goal[4:7]
        
        # Calculate errors
        time_error = abs(current_time - target_time)
        pos_error = np.linalg.norm(paddle_pos - target_pos)
        vel_error = np.linalg.norm(paddle_vel - target_vel)
        
        if current_time < target_time:
            # Approaching target
            time_remaining = target_time - current_time
            
            if time_remaining > 0:
                pos_to_go = target_pos - paddle_pos
                desired_vel = pos_to_go / time_remaining
                
                vel_norm = np.linalg.norm(paddle_vel)
                desired_norm = np.linalg.norm(desired_vel)
                
                if vel_norm > 0 and desired_norm > 0:
                    vel_alignment = np.dot(paddle_vel, desired_vel) / (vel_norm * desired_norm)
                    vel_alignment = max(0, vel_alignment)
                else:
                    vel_alignment = 0
                
                # Progress reward
                if hasattr(self, '_initial_paddle_pos'):
                    initial_distance = np.linalg.norm(target_pos - self._initial_paddle_pos)
                    current_progress = 1.0 - (pos_error / max(0.1, initial_distance))
                else:
                    current_progress = 1.0 - min(1.0, pos_error / 1.0)
                
                reward = (
                    self.position_weight * current_progress +
                    self.velocity_weight * vel_alignment
                )
            else:
                reward = 0.0
            
            goal_achieved = False
            
        else:
            # At or past target time
            success_thresholds = (
                time_error < 0.02 and
                pos_error < 0.05 and
                vel_error < 0.5
            )
            
            if success_thresholds:
                reward = self.success_bonus - pos_error - 0.5 * vel_error
                goal_achieved = True
            else:
                reward = -pos_error - 0.5 * vel_error - time_error
                goal_achieved = False
        
        info = {
            'goal_achieved': goal_achieved,
            'time_error': float(time_error),
            'position_error': float(pos_error),
            'velocity_error': float(vel_error),
            'current_time': float(current_time),
            'target_time': float(target_time)
        }
        
        return float(reward), info
    
    def _describe_goal(self, goal: np.ndarray) -> str:
        """Describe goal in human terms."""
        target_time = goal[0]
        target_pos = goal[1:4]
        target_vel = goal[4:7]
        
        speed = np.linalg.norm(target_vel)
        
        if speed < 1.0:
            swing = "gentle"
        elif speed < 3.0:
            swing = "moderate"
        else:
            swing = "powerful"
        
        return f"Reach {target_pos.round(2)} at t={target_time:.2f}s with {swing} swing"
    
    # Gymnasium interface methods
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()
    
    @property
    def unwrapped(self):
        return self.env
    
    def __getattr__(self, name):
        return getattr(self.env, name)