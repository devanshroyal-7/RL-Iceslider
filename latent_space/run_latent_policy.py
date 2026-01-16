import argparse
import sys
from pathlib import Path
from typing import Set

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "agent"))
sys.path.insert(0, str(PROJECT_ROOT))

from agent.wrappers import make_iceslider_env  # noqa: E402
import ppo_model  # noqa: F401,E402
from latent_action_tracker import LatentStateTracker  # noqa: E402
from models import Encoder  # noqa: E402


def get_action_name(action: int) -> str:
    names = ["UP", "RIGHT", "LEFT", "DOWN", "NOOP"]
    if 0 <= action < len(names):
        return names[action]
    return f"UNKNOWN({action})"


def select_next_best_action(action_probs: torch.Tensor, visited_actions: Set[int]) -> int:
    """
    Select the highest-probability action not yet tried in this latent state.
    """
    print(action_probs)
    probs = action_probs.cpu().numpy().flatten()
    print(probs)
    sorted_actions = np.argsort(probs)[::-1]
    print(sorted_actions)
    for a in sorted_actions:
        if int(a) not in visited_actions:
            return int(a)
    return int(sorted_actions[0])


def prepare_encoder_input(obs: np.ndarray) -> torch.Tensor:
    """
    Prepare observation for encoder (single 84x84 grayscale frame).
    obs shape: (1, 1, 84, 84) from VecTransposeImage (NCHW).
    """
    frame = obs[0, 0, :, :]
    frame_tensor = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).float()
    frame_tensor = frame_tensor / 255.0
    frame_tensor = (frame_tensor - 0.5) / 0.5
    return frame_tensor


def run_latent_policy(
    policy_path: str = str(PROJECT_ROOT / "agent" / "checkpoints_2" / "ppo_iceslider_interrupted.zip"),
    encoder_path: str = str(BASE_DIR / "encoder_model_grayscale.pth"),
    num_episodes: int = 5,
    render: bool = False,
    start_tracking_step: int = 0,
    start_seed: int = 0,
    n_seeds: int = 1000,
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

    print(f"Using seed range: [{start_seed}, {start_seed + n_seeds})")

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

    try:
        for ep in range(num_episodes):
            tracker.reset()
            obs = vec_env.reset()
            done = False
            step = 0
            ep_reward = 0.0
            ep_next_best = 0
            tracking_started = False

            print(f"\nEpisode {ep + 1}/{num_episodes}")

            while not done and step < 10000:
                if not tracking_started and step >= start_tracking_step:
                    tracking_started = True

                encoder_input = prepare_encoder_input(obs).to(device)
                with torch.no_grad():
                    latent_vector = encoder(encoder_input)

                visited_actions = tracker.get_visited_actions(latent_vector) if tracking_started else set()

                action, _ = model.predict(obs, deterministic=True)
                # print(action)
                best_action = int(action[0])

                if tracking_started and best_action in visited_actions:
                    with torch.no_grad():
                        obs_tensor = torch.as_tensor(obs, device=device)
                        dist = model.policy.get_distribution(obs_tensor)
                        action_probs = dist.distribution.probs
                        print(action_probs)
                    action_to_take = select_next_best_action(action_probs, visited_actions)
                    ep_next_best += 1

                    latent_key = tracker.get_latent_key(latent_vector)
                    prev_actions = ", ".join(get_action_name(a) for a in sorted(visited_actions))
                    print(f"[LATENT LOOP] taken={prev_actions}, "
                          f"policy={get_action_name(best_action)}, switching={get_action_name(action_to_take)}")
                else:
                    action_to_take = best_action

                if tracking_started:
                    tracker.record_action(latent_vector, action_to_take)

                obs, rewards, dones, infos = vec_env.step([action_to_take])
                done = bool(dones[0])
                ep_reward += float(rewards[0])
                step += 1

                if render:
                    # VecTransposeImage already provides image-aligned observation;
                    # relying on vec_env.render may be slow but is optional.
                    vec_env.render()

            episode_rewards.append(ep_reward)
            total_next_best_actions += ep_next_best
            print(f"Episode {ep + 1} finished: reward={ep_reward:.2f}, steps={step}, "
                  f"next_best_actions={ep_next_best}")

    finally:
        vec_env.close()

    avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    print(f"\nEvaluation complete: avg reward {avg_reward:.2f} over {num_episodes} episodes")
    print(f"Total next-best actions: {total_next_best_actions} "
          f"(avg {total_next_best_actions/num_episodes if num_episodes else 0:.2f}/episode)")
    return episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Run PPO with latent loop avoidance on IceSlider")
    parser.add_argument(
        "--policy",
        type=str,
        default=str(PROJECT_ROOT / "agent" / "checkpoints_2" / "ppo_iceslider_interrupted.zip")
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=str(BASE_DIR / "encoder_model_grayscale.pth")
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--start-tracking-step", type=int, default=0)
    parser.add_argument("--start-seed", type=int, default=0,
                        help="Starting seed value for evaluation (default: 0)")
    parser.add_argument("--n-seeds", type=int, default=1000,
                        help="Number of seeds in the range [start_seed, start_seed + n_seeds) (default: 1000)")
    args = parser.parse_args()

    run_latent_policy(
        policy_path=args.policy,
        encoder_path=args.encoder,
        num_episodes=args.episodes,
        render=args.render,
        start_tracking_step=args.start_tracking_step,
        start_seed=args.start_seed,
        n_seeds=args.n_seeds,
    )


if __name__ == "__main__":
    main()

