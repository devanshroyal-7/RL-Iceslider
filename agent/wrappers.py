import cv2
import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecTransposeImage
from puzzlegen.ice_slider import IceSlider


class WarpFrame(gym.ObservationWrapper):
    """
    Warp frames to a fixed size (84x84) and grayscale.
    Restored from original code.
    """
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=np.uint8,
        )

    def observation(self, obs):
        assert obs is not None
        # Convert to grayscale if needed
        if obs.ndim == 3 and obs.shape[-1] == 3:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        elif obs.ndim == 2:
            pass
        else:
            obs = np.squeeze(obs)
        resized = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        resized = np.expand_dims(resized.astype(np.uint8), axis=-1)
        return resized

class FixedSeedSampler(gym.Wrapper):
    """
    Forces the environment to sample from a fixed set of seeds.
    Tracks and exposes the current seed in info dict for logging.
    """
    def __init__(self, env, n_seeds=1000, start_seed=0):
        super().__init__(env)
        # Create a pool of specific seeds (e.g., 0 to 999)
        self.seeds = list(range(start_seed, start_seed + n_seeds))
        self.current_seed = None

    def reset(self, **kwargs):
        # Pick a random seed from our training set
        seed = int(np.random.choice(self.seeds))
        self.current_seed = seed
        # Remove seed from kwargs if present to avoid duplicate argument error
        kwargs.pop('seed', None)
        obs, info = self.env.reset(seed=seed, **kwargs)
        info = dict(info)  # Ensure mutable
        info['seed'] = seed
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)  # Ensure mutable
        info['seed'] = self.current_seed
        return obs, reward, terminated, truncated, info

class TimePenaltyWrapper(gym.Wrapper):
    """
    1. Adds step penalty (-0.02).
    2. Adds goal reward (+10.0).
    3. REPLACES the reshaping wrapper.
    Preserves seed info from underlying environment.
    """
    def __init__(self, env, step_penalty=-0.02, goal_reward=10.0):
        super().__init__(env)
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward

    def step(self, action):
        if isinstance(action, (np.ndarray, list)):
            action = int(action[0])
        else:
            action = int(action)

        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Ensure info is mutable and preserve seed
        info = dict(info)
        
        # If env says we are done and reward is positive, we hit the goal
        if (terminated or truncated) and reward > 0:
            reward = self.goal_reward
        else:
            reward = self.step_penalty
            
        return obs, reward, terminated, truncated, info

def make_iceslider_env(
    rank: int, 
    n_seeds: int = 1000,
    start_seed: int = 0,
    max_steps: int = 64,  # <-- NEW: Hard limit
    render_style: str = "grid_world"
):
    def _init():
        env = IceSlider(render_style=render_style)
        
        # 1. Force the strict time limit FIRST
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
        
        # 2. Constrain to specified seed range
        env = FixedSeedSampler(env, n_seeds=n_seeds, start_seed=start_seed)
        
        # 3. Apply our custom reward logic (Step Penalty)
        env = TimePenaltyWrapper(env)
        
        # 4. Visual Preprocessing (Image wrappers)
        env = WarpFrame(env, width=84, height=84)
        
        return env
    return _init

def make_vec_iceslider_env(
    num_envs: int = 64,
    n_seeds: int = 1000, # Train on 1000 levels only
    start_seed: int = 0,
    use_subproc: bool = True,
    render_style: str = "grid_world",
):
    env_fns = [
        make_iceslider_env(rank=i, n_seeds=n_seeds, start_seed=start_seed, max_steps=64, render_style=render_style) 
        for i in range(num_envs)
    ]
    
    VecEnvClass = SubprocVecEnv if use_subproc else DummyVecEnv
    vec_env = VecEnvClass(env_fns)
    vec_env = VecMonitor(vec_env)
    vec_env = VecTransposeImage(vec_env)
    return vec_env