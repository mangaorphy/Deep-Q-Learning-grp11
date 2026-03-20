"""
Play Script - Load and Evaluate Trained DQN Agent
==================================================
Loads a trained DQN model and plays episodes with the agent in Atari Tennis-v5.
Uses GreedyQPolicy (deterministic=True) to maximize performance.

Usage:
    python play.py --model models/exp1_optimizedbase_mlp.zip --episodes 3
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import gymnasium as gym
import ale_py

# Force CPU (same as train.py)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

gym.register_envs(ale_py)

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

ENV_NAME = "ALE/Tennis-v5"


def make_env_for_play(seed: int = 42, render: bool = True):
    """
    Build the Atari Tennis env with AtariPreprocessing + FrameStack.
    Same setup as in train.py for consistency.
    """
    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode, frameskip=1)
    env.reset(seed=seed)
    env = Monitor(env, info_keywords=("lives",))
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=1,
        screen_size=84,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=True,
    )
    env = gym.wrappers.FrameStack(env, num_stack=4)
    return env


# ============================================================================
# PLAY FUNCTION
# ============================================================================

def play_episode(model: DQN, env: gym.Env, episode_num: int = 1, deterministic: bool = True):
    """
    Play one episode with the trained model.
    
    Args:
        model: Trained DQN model
        env: Gymnasium environment
        episode_num: Episode number for display
        deterministic: If True, use GreedyQPolicy (select highest Q-value action)
    
    Returns:
        dict with episode statistics
    """
    obs, info = env.reset()
    done = False
    truncated = False
    episode_reward = 0.0
    episode_steps = 0

    print(f"\n{'='*70}")
    print(f"  EPISODE {episode_num}")
    print(f"  Policy: {'GreedyQPolicy (deterministic)' if deterministic else 'Stochastic'}")
    print(f"{'='*70}")

    while not (done or truncated):
        # Use deterministic=True for GreedyQPolicy (greedy action selection)
        action, _states = model.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        episode_steps += 1

    env.render()  # Render final frame
    
    result = {
        "episode": episode_num,
        "total_reward": episode_reward,
        "steps": episode_steps,
        "average_reward_per_step": episode_reward / max(episode_steps, 1),
    }
    
    print(f"  ✓ Episode Complete")
    print(f"    Total Reward: {episode_reward:.2f}")
    print(f"    Steps: {episode_steps}")
    print(f"    Avg Reward/Step: {episode_reward / max(episode_steps, 1):.4f}")
    print(f"{'='*70}")
    
    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Load and play with a trained DQN agent on Atari Tennis-v5"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/exp1_optimizedbase_mlp.zip",
        help="Path to trained DQN model (default: models/exp1_optimizedbase_mlp.zip)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to play (default: 3)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use GreedyQPolicy (deterministic) for action selection (default: True)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering (no GUI display)",
    )

    args = parser.parse_args()

    # ---- Validate model path -----------------------------------------------
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"✗ Error: Model not found at {model_path}")
        print("\n  Available models:")
        models_dir = Path("models")
        if models_dir.exists():
            for f in sorted(models_dir.glob("*.zip")):
                print(f"    - {f.relative_to('.')}")
        sys.exit(1)

    # ---- Load model --------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  LOADING MODEL")
    print(f"{'='*70}")
    print(f"  Model path: {model_path}")

    try:
        model = DQN.load(str(model_path))
        print(f"  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        sys.exit(1)

    # ---- Create environment ------------------------------------------------
    print(f"\n  Creating environment: {ENV_NAME}")
    render = not args.no_render
    env = make_env_for_play(seed=42, render=render)
    print(f"  ✓ Environment created")
    print(f"{'='*70}")

    # ---- Play episodes -----------------------------------------------------
    all_results = []
    for ep in range(1, args.episodes + 1):
        result = play_episode(model, env, episode_num=ep, deterministic=args.deterministic)
        all_results.append(result)

    # ---- Summary -----------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  PLAY SUMMARY ({args.episodes} episodes)")
    print(f"{'='*70}")
    rewards = [r["total_reward"] for r in all_results]
    print(f"  Mean Reward:   {np.mean(rewards):.2f}")
    print(f"  Max Reward:    {np.max(rewards):.2f}")
    print(f"  Min Reward:    {np.min(rewards):.2f}")
    print(f"  Std Deviation: {np.std(rewards):.2f}")
    print(f"{'='*70}\n")

    env.close()


if __name__ == "__main__":
    main()
