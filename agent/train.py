import argparse
import os
import signal
import sys
from pathlib import Path
from collections import defaultdict

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from wrappers import make_vec_iceslider_env
from ppo_model import make_policy_kwargs


class ManualSaveCallback(BaseCallback):
    """Save model on interruption (Ctrl+C)."""
    
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.model = None
        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        if self.model is not None:
            print("\n[!] Interruption detected. Saving model...")
            try:
                self.model.save(str(self.save_path))
                print(f"[OK] Model saved to {self.save_path}.zip")
            except Exception as e:
                print(f"[!] Failed to save model: {e}")
        sys.exit(0)
    
    def _on_training_start(self):
        # Get model reference from the training environment
        self.model = self.model  # Will be set manually
    
    def _on_step(self) -> bool:
        return True


class SeedLoggerCallback(BaseCallback):
    """Log seed values from episode info to TensorBoard."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_seeds = []
        self.seed_counts = defaultdict(int)
        
    def _on_step(self) -> bool:
        # Get info from all environments
        if hasattr(self.locals, 'infos') and self.locals.get('infos'):
            for info in self.locals['infos']:
                if isinstance(info, dict) and 'seed' in info:
                    seed = info['seed']
                    self.episode_seeds.append(seed)
                    self.seed_counts[seed] += 1
        
        # Log periodically (every 1000 steps)
        if self.n_calls % 1000 == 0 and self.episode_seeds:
            if self.logger is not None:
                # Log unique seeds seen in recent episodes
                recent_seeds = self.episode_seeds[-100:] if len(self.episode_seeds) >= 100 else self.episode_seeds
                unique_seeds = len(set(recent_seeds))
                self.logger.record('episode/unique_seeds_recent', unique_seeds)
                
                # Log total unique seeds seen so far
                total_unique = len(set(self.episode_seeds))
                self.logger.record('episode/unique_seeds_total', total_unique)
                
                # Log most common seed (for debugging)
                if self.seed_counts:
                    most_common_seed = max(self.seed_counts.items(), key=lambda x: x[1])[0]
                    self.logger.record('episode/most_common_seed', float(most_common_seed))
            
            # Clear old data periodically to avoid memory buildup
            if len(self.episode_seeds) > 10000:
                self.episode_seeds = self.episode_seeds[-5000:]
                self.seed_counts.clear()
        
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on IceSlider with Nature CNN")
    parser.add_argument("--total-timesteps", type=int, default=50_000_000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=32, help="Number of parallel environments (reduced from 64 to avoid memory issues)")
    parser.add_argument("--n-steps", type=int, default=256, help="Rollout steps per environment")
    parser.add_argument("--batch-size", type=int, default=1024, help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO epochs")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--logdir", type=str, default="runs", help="Tensorboard log directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_2", help="Directory to store checkpoints")
    parser.add_argument("--checkpoint-freq", type=int, default=1_000_000, help="Save every N steps")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda", help="Training device (cuda/cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Auto-detect device: use CUDA if available and requested, otherwise CPU
    try:
        if args.device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA requested but not available. Falling back to CPU.")
            print(f"   CUDA available: {torch.cuda.is_available()}")
            print(f"   If you have a GPU, you may need to install PyTorch with CUDA support:")
            print(f"   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            device = "cpu"
        elif args.device == "cuda":
            device = "cuda"
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = args.device
            print(f"Using device: {device}")
    except OSError as e:
        if "paging file" in str(e).lower() or "1455" in str(e):
            print("❌ ERROR: Windows paging file is too small to load CUDA libraries.")
            print("   This happens when Windows virtual memory is insufficient.")
            print("\n   Solutions:")
            print("   1. Increase Windows paging file size:")
            print("      - Open System Properties > Advanced > Performance Settings")
            print("      - Virtual Memory > Change > Set to 'System managed size' or at least 16GB")
            print("      - Restart computer after changes")
            print("   2. Install Visual C++ Redistributable:")
            print("      Download: https://aka.ms/vs/16/release/vc_redist.x64.exe")
            print("   3. Reduce memory usage by using fewer parallel environments:")
            print(f"      python train.py --n-envs 16 --device cpu  # Use CPU with fewer envs")
            print("\n   Falling back to CPU for now...")
            device = "cpu"
        else:
            raise

    vec_env = make_vec_iceslider_env(
        num_envs=args.n_envs,
        n_seeds=10000,  # Train on 10000 different seeds
        use_subproc=True,
        render_style="grid_world",
    )

    policy_kwargs = make_policy_kwargs()

    if args.resume:
        model = PPO.load(
            args.resume,
            env=vec_env,
            device=device,
        )
    else:
        model = PPO(
            "CnnPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=args.logdir,
            device=torch.device(device),
            verbose=1,
        )

    # Fix checkpoint frequency: convert timesteps to steps
    # save_freq in CheckpointCallback is in "steps" (number of times _on_step is called)
    # With n_envs environments, each step = n_envs timesteps
    steps_per_save = max(args.checkpoint_freq // args.n_envs, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=steps_per_save,
        save_path=args.checkpoint_dir,
        name_prefix="ppo_iceslider",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    seed_callback = SeedLoggerCallback()
    
    # Manual save callback for interruption handling
    manual_save_path = Path(args.checkpoint_dir) / "ppo_iceslider_interrupted"
    manual_save_callback = ManualSaveCallback(manual_save_path, verbose=1)
    manual_save_callback.model = model  # Set model reference

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, seed_callback, manual_save_callback],
            reset_num_timesteps=not bool(args.resume),
            tb_log_name="ppo_iceslider",
        )
    except KeyboardInterrupt:
        # Fallback: try to save if signal handler didn't work
        print("\n[!] Training interrupted. Attempting to save model...")
        try:
            final_path = Path(args.checkpoint_dir) / "ppo_iceslider_interrupted"
            model.save(str(final_path))
            print(f"[OK] Model saved to {final_path}.zip")
        except Exception as e:
            print(f"[!] Failed to save model: {e}")
        raise

    final_path = Path(args.checkpoint_dir) / "ppo_iceslider_final"
    model.save(str(final_path))
    vec_env.close()


if __name__ == "__main__":
    main()

