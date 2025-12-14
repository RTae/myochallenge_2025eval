import numpy as np
from typing import Tuple, Dict, Optional, Any
from loguru import logger

from myosuite.utils import gym as myo_gym
from config import Config

class TableTennisManager(myo_gym.Env):
    """
    Manager that learns to predict interception targets.
    Now accepts both Worker environment AND trained Worker model.
    """
    
    def __init__(self, 
                 worker_env: Any,        # Worker environment instance
                 worker_model: Any,      # Trained Worker model (PPO)
                 config: Config):
        """
        Initialize Manager with Worker environment and model.
        
        Args:
            worker_env: Worker environment instance
            worker_model: Trained Worker model (PPO)
            config: Configuration
        """
        super().__init__()
        
        self.config = config
        self.worker_env = worker_env
        self.worker_model = worker_model
        
        # MyoChallenge timing
        self.env_time_step = 0.01
        self.max_episode_steps = config.episode_len  # 300
        
        # Manager decision frequency
        self.manager_decision_interval = 10  # Every 0.1s
        self.worker_steps_per_decision = self.manager_decision_interval
        self.max_manager_decisions = self.max_episode_steps // self.manager_decision_interval
        
        # Observation and action spaces
        self.observation_dim = 17
        self.action_dim = 7
        
        self.observation_space = myo_gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )
        
        self.action_space = myo_gym.spaces.Box(
            low=self.worker_env.goal_low,
            high=self.worker_env.goal_high,
            dtype=np.float32
        )
        
        # Tracking
        self.current_step = 0
        self.current_decision = 0
        self.total_hits = 0
        self.total_manager_episodes = 0
        
        # Physics constants
        self.gravity = np.array([0, 0, -9.8])
        self.net_height = 1.59
        
        logger.info(f"✓ Manager initialized")
        logger.info(f"  Worker model: Available")
        logger.info(f"  Max decisions per episode: {self.max_manager_decisions}")
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the hierarchical environment."""
        # Reset Worker environment
        self.worker_env.reset(seed=seed)
        
        # Reset tracking
        self.current_step = 0
        self.current_decision = 0
        self.total_manager_episodes += 1
        
        # Get initial Manager observation
        manager_obs = self._extract_manager_observation()
        
        info = {
            'manager_episode': self.total_manager_episodes,
            'total_hits': self.total_hits,
            'current_step': self.current_step,
            'current_decision': self.current_decision
        }
        
        return manager_obs, info
    
    def step(self, manager_action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute Manager's decision using trained Worker model.
        """
        # 1. Parse Manager's action
        target_time = float(manager_action[0])
        target_pos = manager_action[1:4]
        target_vel = manager_action[4:7]
        
        # 2. Set this as Worker's current goal
        self.worker_env.current_goal = manager_action.copy()
        
        # 3. Execute Worker using trained model
        worker_reward_total = 0.0
        hit_occurred = False
        contact_info = {}
        
        for i in range(self.worker_steps_per_decision):
            # Update environment step counter
            self.current_step += 1
            
            # Get Worker's observation
            worker_obs = self.worker_env._augment_observation()
            
            # Use trained Worker model to get action
            worker_action, _ = self.worker_model.predict(worker_obs, deterministic=True)
            
            # Execute Worker step
            _, worker_reward, worker_terminated, worker_truncated, worker_info = \
                self.worker_env.env.step(worker_action)
            
            worker_reward_total += worker_reward
            
            # Check for ball contact
            obs_dict = self.worker_env.env.obs_dict
            if 'touching_info' in obs_dict and obs_dict['touching_info'][0] > 0.5:
                if not hit_occurred:  # Only count first hit
                    hit_occurred = True
                    contact_info = {
                        'hit_time': obs_dict['time'],
                        'hit_position': obs_dict['paddle_pos'].copy(),
                        'hit_velocity': obs_dict['paddle_vel'].copy(),
                        'step': self.current_step
                    }
                    self.total_hits += 1
            
            # Check if episode should end
            if worker_terminated or worker_truncated:
                break
        
        # 4. Update Manager decision counter
        self.current_decision += 1
        
        # 5. Calculate Manager's reward
        manager_reward = self._calculate_manager_reward(
            hit_occurred=hit_occurred,
            worker_reward_total=worker_reward_total,
            manager_action=manager_action,
            contact_info=contact_info
        )
        
        # 6. Check termination conditions
        terminated = worker_terminated
        truncated = (
            self.current_step >= self.max_episode_steps or
            self.current_decision >= self.max_manager_decisions or
            worker_truncated
        )
        
        # 7. Get next Manager observation (if episode continues)
        if not terminated and not truncated:
            next_manager_obs = self._extract_manager_observation()
        else:
            next_manager_obs = np.zeros(self.observation_dim, dtype=np.float32)
        
        # 8. Prepare info
        info = {
            'manager_step': self.current_step,
            'manager_decision': self.current_decision,
            'max_decisions': self.max_manager_decisions,
            'manager_reward': float(manager_reward),
            'worker_reward_total': float(worker_reward_total),
            'hit_occurred': hit_occurred,
            'total_hits': self.total_hits,
            'target_time': target_time,
            'target_pos': target_pos.tolist(),
            'target_vel': target_vel.tolist(),
            'worker_terminated': worker_terminated,
            'worker_truncated': worker_truncated
        }
        
        if hit_occurred:
            info.update({
                'hit_time': contact_info.get('hit_time', 0),
                'hit_position': contact_info.get('hit_position', [0, 0, 0]),
                'hit_velocity': contact_info.get('hit_velocity', [0, 0, 0]),
                'hit_step': contact_info.get('step', 0)
            })
        
        return next_manager_obs, float(manager_reward), terminated, truncated, info
    
    def _extract_manager_observation(self) -> np.ndarray:
        """
        Extract Manager's observation from the environment.
        """
        obs_dict = self.worker_env.env.obs_dict
        
        # Extract basic state
        ball_pos = obs_dict['ball_pos']
        ball_vel = obs_dict['ball_vel']
        paddle_pos = obs_dict['paddle_pos']
        paddle_vel = obs_dict['paddle_vel']
        current_time = obs_dict['time']
        
        # Calculate derived features
        # 1. Ball height ratio (0-2, where 1.0 = net height)
        ball_height_ratio = ball_pos[2] / self.net_height
        
        # 2. Distance from paddle to ball
        distance_to_ball = np.linalg.norm(ball_pos - paddle_pos)
        
        # 3. Time until ball crosses net (simplified linear prediction)
        net_x = -0.2
        if ball_vel[0] != 0:
            time_to_net = (net_x - ball_pos[0]) / ball_vel[0]
            time_to_net = max(0, min(2.0, time_to_net))
        else:
            time_to_net = 2.0
        
        # 4. Ball direction angles
        ball_speed = np.linalg.norm(ball_vel)
        if ball_speed > 0.1:
            horizontal_angle = np.arctan2(ball_vel[1], ball_vel[0])
            vertical_angle = np.arctan2(ball_vel[2], np.linalg.norm(ball_vel[:2]))
        else:
            horizontal_angle = 0.0
            vertical_angle = 0.0
        
        # Combine into Manager observation using hstack
        manager_obs = np.hstack([
            ball_pos,                    # 0-2: ball position
            ball_vel,                    # 3-5: ball velocity
            paddle_pos,                  # 6-8: paddle position
            paddle_vel,                  # 9-11: paddle velocity
            current_time,                # 12: current time (scalar)
            ball_height_ratio,           # 13: ball height relative to net
            distance_to_ball,            # 14: distance to ball
            time_to_net,                 # 15: time to net crossing
            [horizontal_angle, vertical_angle]  # 16-17: ball angles
        ]).astype(np.float32)
        
        return manager_obs
    
    def _calculate_manager_reward(self,
                                 hit_occurred: bool,
                                 worker_reward_total: float,
                                 manager_action: np.ndarray,
                                 contact_info: Dict) -> float:
        """
        Calculate Manager's reward based on hitting success.
        """
        reward = 0.0
        
        # 1. PRIMARY REWARD: Successful hit
        if hit_occurred:
            reward += 100.0
            
            # Bonus for hitting at the right time
            target_time = manager_action[0]
            hit_time = contact_info.get('hit_time', target_time)
            timing_error = abs(hit_time - target_time)
            
            if timing_error < 0.02:  # Within 20ms
                reward += 20.0
            elif timing_error < 0.05:  # Within 50ms
                reward += 10.0
                
        else:
            # 2. PENALTY for missing
            reward -= 10.0
            
            # Additional penalty based on how close we were
            obs_dict = self.worker_env.env.obs_dict
            ball_pos = obs_dict['ball_pos']
            paddle_pos = obs_dict['paddle_pos']
            distance = np.linalg.norm(ball_pos - paddle_pos)
            
            if distance < 0.1:  # Very close miss
                reward -= 5.0
            elif distance < 0.3:  # Moderate miss
                reward -= 10.0
            else:  # Far miss
                reward -= 20.0
        
        # 3. Incorporate Worker's performance
        reward += worker_reward_total * 0.1
        
        # 4. Penalize unrealistic targets
        target_time = manager_action[0]
        current_time = self.worker_env.env.obs_dict['time']
        
        if target_time < current_time:  # Target in the past
            reward -= 30.0
        elif target_time > current_time + 1.0:  # Too far in future
            reward -= 5.0 * (target_time - current_time - 1.0)
        
        return reward
    
    def render(self):
        """Render the environment."""
        return self.worker_env.render()
    
    def close(self):
        """Close the environment."""
        return self.worker_env.close()
    
    @property
    def unwrapped(self):
        """Get the base environment."""
        return self
    
    def __getattr__(self, name):
        """Forward any other attributes to the Worker environment."""
        return getattr(self.worker_env, name)