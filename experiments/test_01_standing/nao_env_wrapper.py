"""
Wrapper that recreates the environment on fatal physics errors
"""

import numpy as np
import gymnasium as gym

class RecoveryWrapper(gym.Wrapper):
    """
    Detects fatal physics errors and recreates the entire environment
    """
    
    def __init__(self, env_fn):
        """
        Args:
            env_fn: Function that creates a fresh environment
        """
        self.env_fn = env_fn
        self._env = env_fn()
        super().__init__(self._env)
        
        self.recreation_count = 0
        self.max_recreations = 5  # Prevent infinite loop
        
    def reset(self, **kwargs):
        """Reset with recreation on fatal error"""
        try:
            obs, info = self.env.reset(**kwargs)
            return obs, info
        except Exception as e:
            print(f"\n❌ Fatal error during reset: {e}")
            return self._recreate_and_reset(**kwargs)
    
    def step(self, action):
        """Step with recreation on fatal error"""
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Check if physics broke
            if info.get('physics_broken', False):
                print(f"\n⚠️ Physics broken detected, recreating environment...")
                return self._recreate_and_step(action)
            
            return obs, reward, terminated, truncated, info
            
        except Exception as e:
            print(f"\n❌ Fatal error during step: {e}")
            return self._recreate_and_step(action)
    
    def _recreate_and_reset(self, **kwargs):
        """Recreate environment and reset"""
        if self.recreation_count >= self.max_recreations:
            print(f"\n❌ Too many recreations ({self.recreation_count}), giving up")
            raise RuntimeError("Environment recreation failed too many times")
        
        print(f"\n🔄 Recreating environment (attempt {self.recreation_count + 1}/{self.max_recreations})...")
        
        # Close broken environment
        try:
            self.env.close()
        except:
            pass
        
        # Create fresh environment
        self._env = self.env_fn()
        self.env = self._env
        self.recreation_count += 1
        
        print(f"✅ Environment recreated successfully")
        
        # Reset new environment
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def _recreate_and_step(self, action):
        """Recreate environment and take safe step"""
        self._recreate_and_reset()
        
        # Return safe fallback for this step
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[51] = 0.5  # Neutral orientation
        
        return obs, -10.0, True, False, {'recreated': True}
    
    def close(self):
        try:
            self.env.close()
        except:
            pass
