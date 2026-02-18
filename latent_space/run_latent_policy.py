import argparse
import sys
import time
import random
from pathlib import Path
from typing import Optional, Set, Tuple

import cv2
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import pdb

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "agent"))
sys.path.insert(0, str(PROJECT_ROOT))

from agent.wrappers import make_iceslider_env  # noqa: E402
import ppo_model  # noqa: F401,E402
from latent_action_tracker_2 import LatentStateTracker  # noqa: E402
from models import Encoder  # noqa: E402

ACTION_NAMES = ["UP", "RIGHT", "LEFT", "DOWN", "NOOP"]
NUM_ACTIONS = len(ACTION_NAMES)

# 8-connected directions (dx, dy) for sqrl path: N, NE, E, SE, S, SW, W, NW
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
) -> Tuple[int, int, int, int, int]:
    """
    Advance sqrl along a directional path: move by stride in (dx, dy) for this step.
    When remaining reaches 0, sample a new direction and a new length in [1, max_segment_length].
    Returns (new_x, new_y, new_dx, new_dy, new_remaining).
    """
    max_xy = grid_size - sqrl_size
    if max_xy <= 0:
        return x, y, dx, dy, remaining

    if remaining <= 0:
        # Pick new direction (8-connected) and segment length
        dx, dy = rng.choice(SQRL_8_DIRECTIONS)
        remaining = rng.randint(1, max(1, max_segment_length))

    # Move by stride in current direction, clamp to bounds
    nx = max(0, min(max_xy, x + dx * stride))
    ny = max(0, min(max_xy, y + dy * stride))
    remaining -= 1
    return nx, ny, dx, dy, remaining


def get_action_name(action: int) -> str:
    if 0 <= action < len(ACTION_NAMES):
        return ACTION_NAMES[action]
    return f"UNKNOWN({action})"


def select_next_best_action(action_probs: torch.Tensor, visited_actions: Set[int], action_count: list[int]) -> int:
    """
    Select the highest-probability action not yet tried in this latent state.
    """
    probs = action_probs.cpu().numpy().flatten()
    sorted_actions = np.argsort(probs)[::-1]
    # for a in sorted_actions:
    #     if int(a) not in visited_actions:
    #         return int(a)
    action_count = np.array(action_count)
    sorted_action_count = action_count[sorted_actions]
    # breakpoint()
    return int(sorted_actions[np.argmin(sorted_action_count)])
# Try np.max


def prepare_encoder_input(obs: np.ndarray) -> torch.Tensor:
    """
    Prepare observation for encoder (single 84x84 grayscale frame).
    obs shape: (1, 1, 84, 84) from VecTransposeImage (NCHW).
    """
    frame = obs[0, 0, :, :]
    # print(frame.shape)
    frame_tensor = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).float()
    frame_tensor = frame_tensor / 255.0
    frame_tensor = (frame_tensor - 0.5) / 0.5
    return frame_tensor

def maybe_render(
    vec_env,
    delay: float = 0.5,
    scale: int = 8,
    squirrel_xy: Optional[Tuple[int, int]] = None, 
    sqrl_size: Optional[int] = None,
):
    """Render the unwrapped IceSlider environment with delay and scaling.
    squirrel_xy: (x, y) in 84x84 obs space; if set, draw the squirrel on the frame.
    """
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
            # Draw squirrel on render if position given (obs is 84x84, render is 64x64)
            if squirrel_xy is not None:
                x84, y84 = squirrel_xy
                x64 = int(x84 * 64 / 84)
                y64 = int(y84 * 64 / 84)
                # Clip to frame bounds
                h, w = frame.shape[:2]
                size64 = max(0, int(sqrl_size * 64 / 84))
                cv2.rectangle(frame, (x64, y64), (x64+size64, y64+size64), (255, 200, 100), -1)

            # Frame is RGB; convert to BGR for OpenCV display
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Make it bigger by scaling (scale=8 means 8x bigger, so 64x64 becomes 512x512)
            height, width = bgr.shape[:2]
            bigger = cv2.resize(bgr, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("IceSlider", bigger)
            cv2.waitKey(1)  # Small wait for window update
            time.sleep(delay)  # Delay between frames


def run_latent_policy(
    policy_path: str = str(PROJECT_ROOT / "agent" / "ppo_iceslider_main.zip"),
    encoder_path: str = str(BASE_DIR / "encoder_model_grayscale.pth"),
    sqrl_size: int = 0,
    sqrl_stride: int = 1,
    sqrl_max_segment: int = 10,
    num_episodes: int = 5,
    render: bool = False,
    start_tracking_step: int = 0,
    start_seed: int = 0,
    n_seeds: int | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    policy_path = Path(policy_path).resolve()
    encoder_path = Path(encoder_path).resolve()

    if not policy_path.is_file():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    if not encoder_path.is_file():
        raise FileNotFoundError(f"Encoder file not found: {encoder_path}")

    print(f"Loading PPO policy from {policy_path}")
    model = PPO.load(policy_path, device=device)
    model.policy.eval()

    print(f"Loading encoder from {encoder_path}")
    encoder = Encoder(latent_dim=16).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=False))
    encoder.eval()

    use_sequential_seeds = n_seeds is not None      # (1) bool
    if use_sequential_seeds:
        print(f"Using seed range: [{start_seed}, {start_seed + n_seeds})")
        num_episodes = min(num_episodes, n_seeds)

    tracker = LatentStateTracker()

    env_fn = make_iceslider_env(
        rank=0,
        n_seeds=n_seeds,
        start_seed=start_seed,
        max_steps=64,
        render_style="grid_world"
    )
    vec_env = DummyVecEnv([env_fn])
    vec_env = VecTransposeImage(vec_env)

    episode_rewards = []
    total_next_best_actions = 0
    num_successful_eps = 0
    num_succ_w_latent = 0

    try:
        for ep in range(num_episodes):
            tracker.reset()
            episode_rng = None
            if use_sequential_seeds:
                vec_env.seed(start_seed + ep)
                episode_rng = random.Random(start_seed + ep)
            obs = vec_env.reset()
            done = False
            step = 0
            ep_reward = 0.0
            ep_next_best = 0
            tracking_started = False
            
            print(f"\nEpisode {ep + 1}/{num_episodes}")

            # Initialize sqrl at start of episode (directional path: direction + length, stride per step)
            sqrlx, sqrly = 0, 0
            sqrl_dx, sqrl_dy, sqrl_remaining = 0, 0, 0
            if sqrl_size != 0:
                grid_max = 84 - sqrl_size
                if grid_max > 0:
                    if episode_rng is not None:
                        sqrlx = episode_rng.randint(0, grid_max)
                        sqrly = episode_rng.randint(0, grid_max)
                    else:
                        sqrlx = random.randint(0, grid_max)
                        sqrly = random.randint(0, grid_max)

            while not done and step < 10000:
                if not tracking_started and step >= start_tracking_step:
                    tracking_started = True
                # print(obs)
                # print(type(obs))
                # print(obs.shape)
                obs_to_enc = obs.copy()
                if sqrl_size != 0:
                    obs_to_enc[0, 0, sqrlx:sqrlx+sqrl_size, sqrly:sqrly+sqrl_size] = 100
                encoder_input = prepare_encoder_input(obs_to_enc).to(device)
                with torch.no_grad():
                    latent_vector = encoder(encoder_input)

                visited_actions, action_count = tracker.get_visited_actions(latent_vector) if tracking_started else set()

                action, _ = model.predict(obs, deterministic=False)         # this needs to be True to compare how sqrl-size affects 
                # print(action)
                # pdb.set_trace()
                best_action = int(action[0])

                if tracking_started and best_action in visited_actions:
                    with torch.no_grad():
                        obs_tensor = torch.as_tensor(obs, device=device)
                        dist = model.policy.get_distribution(obs_tensor)
                        action_probs = dist.distribution.probs
                        # print(action_probs)
                    action_to_take = select_next_best_action(action_probs, visited_actions, action_count)
                    ep_next_best += 1

                    # latent_key = tracker.get_latent_key(latent_vector)
                    prev_actions = ", ".join(get_action_name(a) for a in sorted(visited_actions))
                    print(f"[LATENT LOOP] taken={prev_actions}, "
                          f"policy={get_action_name(best_action)}, switching={get_action_name(action_to_take)}")
                else:
                    action_to_take = best_action

                if tracking_started:
                    tracker.record_action(latent_vector, action_to_take)

                obs, rewards, dones, infos = vec_env.step([action_to_take])
                done = bool(dones[0])
                if rewards == [10.]:
                    num_successful_eps += 1 
                if rewards == [10.] and ep_next_best > 0:
                    num_succ_w_latent += 1
                ep_reward += float(rewards[0])
                step += 1

                if render and sqrl_size != 0:
                    maybe_render(vec_env, delay=0.2, scale=8, squirrel_xy=(sqrlx, sqrly), sqrl_size=sqrl_size)
                elif render:
                    maybe_render(vec_env, delay=0.2, scale=8)

                # Advance sqrl along directional path (stride per step; new direction + length when segment ends)
                if sqrl_size != 0:
                    rng = episode_rng if episode_rng is not None else random
                    sqrlx, sqrly, sqrl_dx, sqrl_dy, sqrl_remaining = sqrl_advance_directional(
                        sqrlx, sqrly, sqrl_dx, sqrl_dy, sqrl_remaining,
                        grid_size=84, sqrl_size=sqrl_size, stride=sqrl_stride,
                        max_segment_length=sqrl_max_segment, rng=rng,
                    )
            episode_rewards.append(ep_reward)
            total_next_best_actions += ep_next_best
            print(f"Episode {ep + 1} finished: reward={ep_reward:.2f}, steps={step}, "
                  f"next_best_actions={ep_next_best}")

    finally:
        vec_env.close()
        if render:
            cv2.destroyAllWindows()

    avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    print(f"\nEvaluation complete: avg reward {avg_reward:.2f} over {num_episodes} episodes")
    print(f"Total next-best actions: {total_next_best_actions} "
          f"(avg {total_next_best_actions/num_episodes if num_episodes else 0:.2f}/episode)\n"
          f"Number of successful episodes: {num_successful_eps} / {num_episodes} with {num_succ_w_latent} using RTHS")
    return episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Run PPO with latent loop avoidance on IceSlider")
    parser.add_argument(
        "--policy",
        type=str,
        default=str(PROJECT_ROOT / "agent" / "ppo_iceslider_main.zip")
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=str(BASE_DIR / "encoder_model_grayscale.pth")
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--sqrl-size", type=int, default=0)
    parser.add_argument("--sqrl-stride", type=int, default=1,
                        help="Grid cells to move per step in current direction (default: 1)")
    parser.add_argument("--sqrl-max-segment", type=int, default=10,
                        help="Max steps in one direction before picking a new direction (default: 10)")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--start-tracking-step", type=int, default=0)
    parser.add_argument("--start-seed", type=int, default=0,
                        help="Starting seed value for evaluation (default: 0)")
    parser.add_argument("--n-seeds", type=int, default=None,
                        help="If set, iterate seeds in [start_seed, start_seed + n_seeds); else sample random seed per episode")
    args = parser.parse_args()

    run_latent_policy(
        policy_path=args.policy,
        encoder_path=args.encoder,
        num_episodes=args.episodes,
        sqrl_size=args.sqrl_size,
        sqrl_stride=args.sqrl_stride,
        sqrl_max_segment=args.sqrl_max_segment,
        render=args.render,
        start_tracking_step=args.start_tracking_step,
        start_seed=args.start_seed,
        n_seeds=args.n_seeds,
    )


if __name__ == "__main__":
    main()

