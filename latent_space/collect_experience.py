"""
Collect pre-processed 84x84 grayscale experience data from IceSlider using PPO.
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Union

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from tqdm import tqdm

# Ensure custom policy modules are importable when loading PPO
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "agent"))

from agent.wrappers import WarpFrame, TimePenaltyWrapper  # noqa: E402
import ppo_model  # noqa: F401,E402 (needed for custom policy kwargs)
from puzzlegen.ice_slider import IceSlider  # noqa: E402

# -- Configuration --
NUM_EPISODES = 20000
OUTPUT_PATH = Path(__file__).resolve().parent / "iceslider_experience.pkl"
POLICY_PATH = PROJECT_ROOT / "agent" / "checkpoints_2" / "ppo_iceslider_interrupted.zip"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 84
MAX_STEPS_PER_EPISODE = 15


def make_collection_env(max_steps: int):
    """
    Build a single IceSlider env mirroring training preprocessing but with
    a shorter episode horizon for data collection.
    """

    def _init():
        env = IceSlider(render_style="grid_world")
        # Cap episode length for faster collection
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
        env = TimePenaltyWrapper(env)
        env = WarpFrame(env, width=IMAGE_SIZE, height=IMAGE_SIZE)
        return env

    return _init


def make_vec_env(max_steps: int):
    env_fns = [make_collection_env(max_steps)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecTransposeImage(vec_env)
    return vec_env


def extract_frame(obs: np.ndarray) -> np.ndarray:
    """
    obs shape: (1, 1, 84, 84) from VecTransposeImage (NCHW).
    Returns single (84, 84) frame.
    """
    return obs[0, 0, :, :].copy()


def collect_experience(
    num_episodes: int = NUM_EPISODES,
    max_steps_per_episode: int = MAX_STEPS_PER_EPISODE,
    output_path: Union[Path, str] = OUTPUT_PATH,
    policy_path: Union[Path, str] = POLICY_PATH,
):
    """
    Runs PPO policy to collect (s_t, s_t1, a_t) tuples.
    """
    print(f"Using device: {DEVICE}")

    policy_path = Path(policy_path)
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    print(f"Loading PPO policy from {policy_path}")
    model = PPO.load(str(policy_path), device=DEVICE)
    model.policy.eval()

    env = make_vec_env(max_steps_per_episode)

    experience_buffer = []

    print(f"Collecting experience from {num_episodes} episodes (max {max_steps_per_episode} steps each)")

    for episode in tqdm(range(num_episodes), desc="Episodes"):
        # Ensure unique seeds per episode: seed = episode index
        env.seed(episode)
        obs = env.reset()
        done = False
        step = 0

        state_frame = extract_frame(obs)

        while not done and step < max_steps_per_episode:
            action, _ = model.predict(obs, deterministic=True)
            action_int = int(action[0])

            next_obs, reward, dones, infos = env.step(action)
            done = bool(dones[0])

            next_state_frame = extract_frame(next_obs)

            experience_buffer.append((state_frame.copy(), next_state_frame.copy(), action_int))

            obs = next_obs
            state_frame = next_state_frame
            step += 1

    env.close()

    print(f"\nCollected {len(experience_buffer)} transitions from {num_episodes} episodes")
    print(f"Avg steps/episode: {len(experience_buffer) / num_episodes:.2f}")

    print(f"\nSaving experience to {output_path}...")
    with open(Path(output_path), "wb") as f:
        pickle.dump(experience_buffer, f)

    print("Experience collection complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect IceSlider experience for latent encoder")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="Number of episodes to collect")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE, help="Max steps per episode")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH), help="Where to save the pickle buffer")
    parser.add_argument("--policy", type=str, default=str(POLICY_PATH), help="Path to PPO checkpoint")
    args = parser.parse_args()

    collect_experience(
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        output_path=args.output,
        policy_path=args.policy,
    )

