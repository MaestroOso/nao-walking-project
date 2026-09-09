"""
NAO Standing Environment - With CSV Logging
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import os
import csv
from datetime import datetime

sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('../..'))

from isaacsim import SimulationApp

class NaoStandingEnv(gym.Env):
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, headless=False, render_mode='human', log_file=None):
        super().__init__()
        
        print("="*80)
        print("NAO STANDING ENVIRONMENT")
        print("="*80)
        
        from nao_constants import JOINT_NAMES
        
        self.controlled_joint_indices = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 27
        ]
        
        self.num_controlled = len(self.controlled_joint_indices)
        self.obs_size = self.num_controlled * 2 + 10
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_size,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_controlled,), dtype=np.float32
        )
        
        self.max_episode_steps = 500
        self.current_step = 0
        self.action_repeat = 5
        
        self.max_action_change = np.deg2rad(2.0)
        
        self.episode_reward = 0.0
        self.episode_length = 0
        self.longest_standing_time = 0
        self.total_episodes = 0
        self.physics_initialized = False
        
        self.episodes_since_physics_reset = 0
        self.PHYSICS_RESET_INTERVAL = 20
        
        self.current_joint_targets = None
        self.SPAWN_HEIGHT = 0.35
        
        # CSV Logging
        self.log_file = log_file
        self.csv_writer = None
        if self.log_file:
            self.csv_file = open(self.log_file, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                'Episode', 'Reward', 'Steps', 'BestEver', 'Timestamp', 'Status'
            ])
            self.csv_file.flush()
            print(f"✓ Logging to: {self.log_file}")
        
        self.simulation_app = SimulationApp({
            "headless": headless,
            "width": 1280,
            "height": 720
        })
        
        from omni.isaac.core import World
        from omni.isaac.core.articulations import Articulation
        import omni.kit.commands
        import isaacsim.asset.importer.urdf._urdf as urdf_module
        
        self.world = World(
            physics_dt=1.0/120.0,
            rendering_dt=1.0/60.0,
        )
        
        self.world.scene.add_default_ground_plane()
        
        config = urdf_module.ImportConfig()
        config.merge_fixed_joints = False
        config.fix_base = False
        config.import_inertia_tensor = True
        config.distance_scale = 1.0
        
        urdf_path = "../../robots/nao_isaac_sim/nao.urdf"
        
        result = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_path,
            import_config=config
        )
        
        self.nao = self.world.scene.add(
            Articulation(result[1], name="nao")
        )
        
        self.world.reset()
        self.physics_initialized = True
        
        from nao_constants import JOINT_LIMITS
        self.joint_limits = JOINT_LIMITS[self.controlled_joint_indices]
        
        print("✅ READY\n")
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.total_episodes += 1
        self.episodes_since_physics_reset += 1
        
        # Periodic physics reset
        if self.episodes_since_physics_reset >= self.PHYSICS_RESET_INTERVAL:
            print(f"\n🔄 Physics reset (Ep {self.total_episodes})...")
            try:
                self.world.reset()
                print("✅ Done\n")
            except Exception as e:
                print(f"⚠️ Failed: {e}")
            
            self.episodes_since_physics_reset = 0
        
        # Reset NAO
        try:
            self.nao.set_world_pose(
                position=np.array([0.0, 0.0, self.SPAWN_HEIGHT]),
                orientation=np.array([0.0, 0.0, 0.0, 1.0])
            )
            
            neutral = np.zeros(self.nao.num_dof)
            self.nao.set_joint_positions(neutral)
            self.nao.set_joint_velocities(np.zeros(self.nao.num_dof))
            
            self.current_joint_targets = neutral.copy()
            
        except Exception as e:
            print(f"⚠️ Reset: {e}")
        
        obs = self._get_obs()
        return obs, {}
    
    def step(self, action):
        try:
            if np.any(np.isnan(action)):
                obs = self._get_safe_obs()
                self._log_episode("NaN_action")
                return obs, -10.0, True, False, {}
            
            delta = action * self.max_action_change
            
            for i, j in enumerate(self.controlled_joint_indices):
                self.current_joint_targets[j] += delta[i]
                self.current_joint_targets[j] = np.clip(
                    self.current_joint_targets[j],
                    self.joint_limits[i][0],
                    self.joint_limits[i][1]
                )
            
            if np.any(np.isnan(self.current_joint_targets)):
                self.current_joint_targets = np.zeros(self.nao.num_dof)
            
            self.nao.set_joint_positions(self.current_joint_targets)
            
            for _ in range(self.action_repeat):
                self.world.step(render=True)
            
            self.current_step += 1
            self.episode_length += 1
            
            obs = self._get_obs()
            
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)) or np.any(np.abs(obs) > 1e6):
                obs = self._get_safe_obs()
                self._log_episode("Invalid_obs")
                return obs, -10.0, True, False, {}
            
            reward, done = self._reward(obs)
            self.episode_reward += reward
            
            truncated = self.current_step >= self.max_episode_steps
            
            if self.episode_length > self.longest_standing_time:
                self.longest_standing_time = self.episode_length
            
            if done or truncated:
                status = "Timeout" if truncated else "Fell"
                self._log_episode(status)
                
                # Only print every 10 episodes to reduce console spam
                if self.total_episodes % 10 == 0:
                    print(f"Ep {self.total_episodes}: R={self.episode_reward:.1f}, "
                          f"L={self.episode_length}, Best={self.longest_standing_time}")
            
            return obs, reward, done, truncated, {}
            
        except Exception as e:
            print(f"⚠️ Step error: {e}")
            obs = self._get_safe_obs()
            self._log_episode(f"Exception_{str(e)[:20]}")
            return obs, -10.0, True, False, {}
    
    def _log_episode(self, status):
        """Log episode data to CSV"""
        if self.csv_writer:
            self.csv_writer.writerow([
                self.total_episodes,
                f"{self.episode_reward:.1f}",
                self.episode_length,
                self.longest_standing_time,
                datetime.now().strftime("%H:%M:%S"),
                status
            ])
            self.csv_file.flush()
    
    def _get_obs(self):
        try:
            jp = self.nao.get_joint_positions()
            jv = self.nao.get_joint_velocities()
            _, quat = self.nao.get_world_pose()
            lv = self.nao.get_linear_velocity()
            av = self.nao.get_angular_velocity()
            
            if jp is None or jv is None or quat is None or lv is None or av is None:
                return self._get_safe_obs()
            
            jp_c = jp[self.controlled_joint_indices]
            jv_c = jv[self.controlled_joint_indices]
            
            obs = np.concatenate([jp_c, jv_c, quat, lv, av]).astype(np.float32)
            
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                return self._get_safe_obs()
            
            obs = np.clip(obs, -1000, 1000)
            return obs
            
        except:
            return self._get_safe_obs()
    
    def _get_safe_obs(self):
        obs = np.zeros(self.obs_size, dtype=np.float32)
        obs[51] = 0.5
        return obs
    
    def _reward(self, obs):
        qw = obs[51]
        upright = abs(qw) > 0.7
        return (1.0, False) if upright else (-10.0, True)
    
    def render(self):
        pass
    
    def close(self):
        if self.csv_writer:
            self.csv_file.close()
            print(f"\n📊 Episode log saved to: {self.log_file}")
        
        try:
            self.simulation_app.close()
        except:
            pass
