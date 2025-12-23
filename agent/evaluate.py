import argparse
import time

import cv2
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage

from wrappers import WarpFrame, TimePenaltyWrapper, FixedSeedSampler
from puzzlegen.ice_slider import IceSlider


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO model on IceSlider")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained PPO model")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render environment with OpenCV")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for evaluation env")
    return parser.parse_args()


def maybe_render(vec_env, delay: float = 0.2, scale: int = 8):
    """Render the unwrapped IceSlider environment with delay and scaling."""
    # Unwrap through VecEnv wrappers to get to the base environment
    base_vec = vec_env
    while hasattr(base_vec, 'venv') or hasattr(base_vec, 'env'):
        if hasattr(base_vec, 'venv'):
            base_vec = base_vec.venv
        elif hasattr(base_vec, 'env'):
            base_vec = base_vec.env
        else:
            break
    
    # Get the first environment from the VecEnv
    if hasattr(base_vec, 'envs') and len(base_vec.envs) > 0:
        base_env = base_vec.envs[0]
        # Unwrap through gym wrappers to get to the base IceSlider
        # The order is: WarpFrame -> TimePenaltyWrapper -> FixedSeedSampler -> TimeLimit -> IceSlider
        unwrapped_env = base_env
        # Keep unwrapping until we find the IceSlider (it has _get_image method)
        while hasattr(unwrapped_env, 'env'):
            # Check if current level is IceSlider
            if hasattr(unwrapped_env, '_get_image'):
                break
            unwrapped_env = unwrapped_env.env
        
        # Try to get the image directly from the base IceSlider
        frame = None
        if hasattr(unwrapped_env, '_get_image'):
            # Call _get_image directly to bypass any wrapper render() methods
            frame = unwrapped_env._get_image()
        elif hasattr(unwrapped_env, 'render'):
            frame = unwrapped_env.render()
        
        if frame is not None:
            # Frame is RGB; convert to BGR for OpenCV display
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Make it bigger by scaling (scale=8 means 8x bigger, so 64x64 becomes 512x512)
            height, width = bgr.shape[:2]
            bigger = cv2.resize(bgr, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("IceSlider", bigger)
            cv2.waitKey(1)  # Small wait for window update
            time.sleep(delay)  # Delay between frames


def make_eval_env(render_style="grid_world", start_seed=10000, n_seeds=1000):
    """Create evaluation environment with seeds starting from start_seed."""
    def _init():
        env = IceSlider(render_style=render_style)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=64)
        # Use seeds starting from start_seed (e.g., 10000-10999 for evaluation)
        env = FixedSeedSampler(env, n_seeds=n_seeds, start_seed=start_seed)
        env = TimePenaltyWrapper(env)
        env = WarpFrame(env, width=84, height=84)
        return env
    return _init

def evaluate(model_path: str, episodes: int, render: bool, seed: int = 0):
    # Create evaluation environment with seeds > 10000
    env_fn = make_eval_env(render_style="grid_world", start_seed=10000, n_seeds=1000)
    vec_env = DummyVecEnv([env_fn])
    vec_env = VecMonitor(vec_env)
    vec_env = VecTransposeImage(vec_env)
    
    model = PPO.load(model_path, env=vec_env, device="cuda")

    returns = []
    for ep in range(episodes):
        obs = vec_env.reset()
        done = False
        ep_reward = 0.0
        episode_seed = 'unknown'

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            done = bool(dones[0])
            ep_reward += rewards[0]
            # Get seed from step info (FixedSeedSampler adds it to info)
            if isinstance(infos, list) and len(infos) > 0 and 'seed' in infos[0]:
                episode_seed = infos[0]['seed']
            if render:
                maybe_render(vec_env, delay=0.2, scale=8)

        returns.append(ep_reward)
        print(f"Episode {ep + 1}/{episodes}: seed={episode_seed}, reward={ep_reward:.2f}")

    vec_env.close()
    if render:
        cv2.destroyAllWindows()

    rewards = np.array(returns, dtype=np.float32)
    print(f"Mean reward: {rewards.mean():.2f} +/- {rewards.std():.2f} over {episodes} episodes")
    return rewards


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model_path, args.episodes, args.render, seed=args.seed)



