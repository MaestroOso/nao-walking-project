"""
Wrapper that recreates environment every N episodes
"""

import numpy as np
import gymnasium as gym

class FreshEnvWrapper(gym.Wrapper):
    """
    Recreates the entire environment every N episodes to prevent physics degradation
    """
    
    def __init__(self, env_fn, recreate_every=10):
        """
        Args:
            env_fn: Function that creates a fresh environment
            recreate_every: Recreate environment every N episodes
        """
        self.env_fn = env_fn
        self.recreate_every = recreate_every
        
        self._env = env_fn()
        super().__init__(self._env)
        
        self.episode_count = 0
        self.total_recreations = 0
        
        print(f"✓ Will recreate environment every {recreate_every} episodes")
    
    def reset(self, **kwargs):
        """Reset with periodic recreation"""
        self.episode_count += 1
        
        # Recreate environment periodically
        if self.episode_count % self.recreate_every == 0:
            print(f"\n🔄 Recreating environment (episode {self.episode_count})...")
            
            try:
                # Close old environment
                self.env.close()
            except Exception as e:
                print(f"   Warning during close: {e}")
            
            # Create fresh environment
            try:
                self._env = self.env_fn()
                self.env = self._env
                self.total_recreations += 1
                print(f"✅ Fresh environment created (recreation #{self.total_recreations})\n")
            except Exception as e:
                print(f"❌ Failed to recreate: {e}")
                raise
        
        # Reset (new or existing environment)
        try:
            obs, info = self.env.reset(**kwargs)
            return obs, info
        except Exception as e:
            print(f"❌ Reset failed: {e}")
            print("   Attempting emergency recreation...")
            
            # Emergency recreation
            try:
                self.env.close()
            except:
                pass
            
            self._env = self.env_fn()
            self.env = self._env
            self.total_recreations += 1
            
            obs, info = self.env.reset(**kwargs)
            return obs, info
    
    def step(self, action):
        """Step with error recovery"""
        try:
            return self.env.step(action)
        except Exception as e:
            print(f"❌ Step failed: {e}")
            print("   Ending episode, will recreate on next reset")
            
            # Return safe observation and end episode
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            obs[51] = 0.5
            return obs, -10.0, True, False, {'step_failed': True}
    
    def close(self):
        try:
            self.env.close()
        except:
            pass
