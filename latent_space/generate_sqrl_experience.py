"""
Generate encoder training data with path-following sqrls.
Runs the IceSlider env, collects (s_t, s_t1, a_t) and overlays a sqrl that
moves each step to an 8-connected adjacent cell. Saves same format as
iceslider_experience.pkl for use with train.py.
"""

import pickle
import random
import sys
from pathlib import Path
from typing import Union

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "agent"))

from agent.wrappers import WarpFrame, TimePenaltyWrapper  # noqa: E402
import ppo_model  # noqa: F401,E402
from puzzlegen.ice_slider import IceSlider  # noqa: E402

# 8-connected directions (dx, dy)
SQRL_8_DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def sqrl_advance_directional(
    x: int, y: int,
    dx: int, dy: int,
    remaining: int,
    grid_size: int,
    sqrl_size: int,
    stride: int,
    max_segment_length: int,
    rng: random.Random,
) -> tuple[int, int, int, int, int]:
    """
    Advance sqrl along a directional path: move by stride in (dx, dy) for this step.
    When remaining reaches 0, sample a new direction and a new length in [1, max_segment_length].
    Returns (new_x, new_y, new_dx, new_dy, new_remaining).
    """
    max_xy = grid_size - sqrl_size
    if max_xy <= 0:
        return x, y, dx, dy, remaining

    if remaining <= 0:
        dx, dy = rng.choice(SQRL_8_DIRECTIONS)
        remaining = rng.randint(1, max(1, max_segment_length))

    nx = max(0, min(max_xy, x + dx * stride))
    ny = max(0, min(max_xy, y + dy * stride))
    remaining -= 1
    return nx, ny, dx, dy, remaining


def overlay_sqrl(frame: np.ndarray, x: int, y: int, sqrl_size: int, value: float = 100.0) -> None:
    """Draw sqrl patch on frame in-place. frame shape (H, W)."""
    if sqrl_size <= 0:
        return
    h, w = frame.shape[:2]
    x1 = min(x + sqrl_size, w)
    y1 = min(y + sqrl_size, h)
    x0 = max(0, x)
    y0 = max(0, y)
    frame[y0:y1, x0:x1] = value


# -- Config --
NUM_EPISODES = 20000
OUTPUT_PATH = Path(__file__).resolve().parent / "iceslider_sqrl_experience.pkl"
POLICY_PATH = PROJECT_ROOT / "agent" / "ppo_iceslider_main.zip"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 84
MAX_STEPS_PER_EPISODE = 15
DEFAULT_SQRL_SIZE = 5
DEFAULT_SQRL_STRIDE = 1
DEFAULT_SQRL_MAX_SEGMENT = 10


def make_collection_env(max_steps: int):
    def _init():
        env = IceSlider(render_style="grid_world")
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
        env = TimePenaltyWrapper(env)
        env = WarpFrame(env, width=IMAGE_SIZE, height=IMAGE_SIZE)
        return env
    return _init


def make_vec_env(max_steps: int):
    vec_env = DummyVecEnv([make_collection_env(max_steps)])
    return VecTransposeImage(vec_env)


def extract_frame(obs: np.ndarray) -> np.ndarray:
    """(1, 1, 84, 84) -> (84, 84)."""
    return obs[0, 0, :, :].copy()


def generate_sqrl_experience(
    num_episodes: int = NUM_EPISODES,
    max_steps_per_episode: int = MAX_STEPS_PER_EPISODE,
    sqrl_size: int = DEFAULT_SQRL_SIZE,
    sqrl_stride: int = DEFAULT_SQRL_STRIDE,
    sqrl_max_segment: int = DEFAULT_SQRL_MAX_SEGMENT,
    output_path: Union[Path, str] = OUTPUT_PATH,
    policy_path: Union[Path, str] = POLICY_PATH,
    start_seed: int = 0,
):
    """
    Collect (s_t, s_t1, a_t) with a path-following sqrl overlaid.
    Sqrl follows directional segments: picks a direction and a length, moves by
    stride each step in that direction, then picks a new direction and length.
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

    print(f"Collecting experience with path-following sqrl (size={sqrl_size}, stride={sqrl_stride}, max_segment={sqrl_max_segment})")
    print(f"Episodes: {num_episodes}, max steps/episode: {max_steps_per_episode}")

    for episode in tqdm(range(num_episodes), desc="Episodes"):
        seed = start_seed + episode
        env.seed(seed)
        rng = random.Random(seed)
        obs = env.reset()
        done = False
        step = 0

        grid_max = IMAGE_SIZE - sqrl_size
        if grid_max <= 0:
            sqrlx, sqrly = 0, 0
        else:
            sqrlx = rng.randint(0, grid_max)
            sqrly = rng.randint(0, grid_max)
        sqrl_dx, sqrl_dy, sqrl_remaining = 0, 0, 0

        state_frame = extract_frame(obs)

        while not done and step < max_steps_per_episode:
            action, _ = model.predict(obs, deterministic=True)
            action_int = int(action[0])

            next_obs, reward, dones, infos = env.step(action)
            done = bool(dones[0])
            next_state_frame = extract_frame(next_obs)

            # Advance sqrl along directional path (stride per step; new direction + length when segment ends)
            next_sqrlx, next_sqrly, next_dx, next_dy, next_remaining = sqrl_advance_directional(
                sqrlx, sqrly, sqrl_dx, sqrl_dy, sqrl_remaining,
                grid_size=IMAGE_SIZE, sqrl_size=sqrl_size, stride=sqrl_stride,
                max_segment_length=sqrl_max_segment, rng=rng,
            )

            s_t_aug = state_frame.copy()
            s_t1_aug = next_state_frame.copy()
            overlay_sqrl(s_t_aug, sqrlx, sqrly, sqrl_size)
            overlay_sqrl(s_t1_aug, next_sqrlx, next_sqrly, sqrl_size)
            experience_buffer.append((s_t_aug, s_t1_aug, action_int))

            obs = next_obs
            state_frame = next_state_frame
            sqrlx, sqrly = next_sqrlx, next_sqrly
            sqrl_dx, sqrl_dy, sqrl_remaining = next_dx, next_dy, next_remaining
            step += 1

    env.close()

    print(f"\nCollected {len(experience_buffer)} transitions from {num_episodes} episodes")
    print(f"Avg steps/episode: {len(experience_buffer) / num_episodes:.2f}")

    output_path = Path(output_path)
    print(f"Saving to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(experience_buffer, f)
    print("Done. Use this file with train.py (e.g. --experience iceslider_sqrl_experience.pkl).")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate encoder data with path-following sqrls")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE)
    parser.add_argument("--sqrl-size", type=int, default=DEFAULT_SQRL_SIZE)
    parser.add_argument("--sqrl-stride", type=int, default=DEFAULT_SQRL_STRIDE,
                        help="Grid cells to move per step in current direction")
    parser.add_argument("--sqrl-max-segment", type=int, default=DEFAULT_SQRL_MAX_SEGMENT,
                        help="Max steps in one direction before picking a new direction")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--policy", type=str, default=str(POLICY_PATH))
    parser.add_argument("--start-seed", type=int, default=0)
    args = parser.parse_args()

    generate_sqrl_experience(
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        sqrl_size=args.sqrl_size,
        sqrl_stride=args.sqrl_stride,
        sqrl_max_segment=args.sqrl_max_segment,
        output_path=args.output,
        policy_path=args.policy,
        start_seed=args.start_seed,
    )
