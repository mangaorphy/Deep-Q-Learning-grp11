"""
Deep Q-Network (DQN) Training Script - Your Experiments
=======================================================
MlpPolicy only | 10 hyperparameter experiments | ALE/Tennis-v5

Each experiment is saved individually to your models/
Results are logged to your logs/ and a summary JSON is written at the end.

Member: Your Name
"""

import os
import sys
import json
import time
import gc
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import gymnasium as gym
import ale_py

# Force CPU on Kaggle (P100 CUDA compatibility issue)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

gym.register_envs(ale_py)

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# ============================================================================
# PATHS  (all outputs stay inside current directory)
# ============================================================================

BASE_DIR   = Path(__file__).parent          # .../Deep-Q-Learning-grp11/
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR   = BASE_DIR / "logs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

ENV_NAME       = "ALE/Tennis-v5"
POLICY_TYPE    = "MlpPolicy"          # Fixed: MLP only
TOTAL_TIMESTEPS_DEFAULT = 300_000     # Default timesteps for full Tennis learning (18 actions)
SEED           = 42

# ============================================================================
# ENVIRONMENT FACTORY
# ============================================================================

def make_env(seed: int = SEED) -> gym.Env:
    """
    Build the Atari Tennis env with AtariPreprocessing + FrameStack.
    Observation shape after wrappers: (4, 84, 84) — SB3 MlpPolicy
    auto-flattens this via FlattenExtractor.
    """
    env = gym.make(ENV_NAME, render_mode=None, frameskip=1)
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
# REWARD CALLBACK
# ============================================================================

class RewardLogger(BaseCallback):
    """Tracks episode rewards and lengths during training."""

    def __init__(self, log_freq: int = 5_000):
        super().__init__()
        self.log_freq       = log_freq
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int]   = []
        self._current_rewards: Dict       = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        return True

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_experiment(
    exp_name: str,
    hyperparams: Dict,
    total_timesteps: int = TOTAL_TIMESTEPS_DEFAULT,
) -> Dict:
    """
    Train one DQN experiment with MlpPolicy and given hyperparameters.

    Returns a dict with results suitable for the summary table.
    """
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT : {exp_name}")
    print(f"  Policy     : {POLICY_TYPE}")
    print(f"  Timesteps  : {total_timesteps:,}")
    print(f"  Params     : {hyperparams}")
    print(f"{'='*70}")

    env = make_env(seed=SEED)

    model = DQN(
        POLICY_TYPE,
        env,
        learning_rate           = hyperparams["learning_rate"],
        gamma                   = hyperparams["gamma"],
        batch_size              = hyperparams["batch_size"],
        buffer_size             = hyperparams.get("buffer_size", 100_000),
        target_update_interval  = hyperparams.get("target_update_interval", 1_000),
        learning_starts         = hyperparams.get("learning_starts", 1_000),
        exploration_fraction    = hyperparams.get("epsilon_decay", 50_000) / total_timesteps,
        exploration_initial_eps = hyperparams.get("epsilon_start", 1.0),
        exploration_final_eps   = hyperparams["epsilon_end"],
        verbose                 = 1,
        device                  = "auto",
        policy_kwargs           = {"normalize_images": False},
    )

    callback = RewardLogger(log_freq=1_000)

    t_start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    elapsed = time.time() - t_start

    # ---- save model --------------------------------------------------------
    safe_name   = exp_name.lower().replace(" ", "_")
    model_path  = MODELS_DIR / f"{safe_name}_mlp.zip"
    model.save(str(model_path))
    print(f"  ✓ Model saved → {model_path}")

    # ---- collect stats -----------------------------------------------------
    rewards = callback.episode_rewards
    result = {
        "experiment"    : exp_name,
        "policy"        : POLICY_TYPE,
        "hyperparams"   : hyperparams,
        "total_timesteps": total_timesteps,
        "episodes"      : len(rewards),
        "mean_reward"   : float(np.mean(rewards))   if rewards else 0.0,
        "max_reward"    : float(np.max(rewards))     if rewards else 0.0,
        "min_reward"    : float(np.min(rewards))     if rewards else 0.0,
        "std_reward"    : float(np.std(rewards))     if rewards else 0.0,
        "training_time" : round(elapsed, 2),
        "model_path"    : str(model_path),
        "episode_rewards": rewards,
    }

    # ---- save individual log -----------------------------------------------
    log_path = LOGS_DIR / f"{safe_name}.json"
    with open(log_path, "w") as f:
        log_data = {k: v for k, v in result.items() if k != "episode_rewards"}
        json.dump(log_data, f, indent=2)
    print(f"  ✓ Log saved  → {log_path}")

    # ---- quick plot (optional) ----------------------------------------------
    if PLOTTING_AVAILABLE and rewards:
        _plot_rewards(rewards, exp_name, safe_name)

    env.close()
    del env
    del model
    gc.collect()  # Force garbage collection to free memory
    time.sleep(1)  # Brief pause for OS to handle freed memory
    return result


def _plot_rewards(rewards: List[float], title: str, safe_name: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rewards, alpha=0.5, color="steelblue", label="Episode reward")
    if len(rewards) >= 20:
        window = min(50, len(rewards) // 5)
        ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), ma,
                color="darkorange", linewidth=2,
                label=f"{window}-ep moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(f"[Your Experiments] {title}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = LOGS_DIR / f"{safe_name}_rewards.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  ✓ Plot saved → {plot_path}")


# ============================================================================
# YOUR 10 EXPERIMENTS  (MlpPolicy only)
# ============================================================================

YOUR_EXPERIMENTS = [
    {
        "name": "Exp1_OptimizedBase",
        "hyperparams": {
            "learning_rate" : 5e-4,
            "gamma"         : 0.99,
            "batch_size"    : 32,
            "buffer_size"   : 500_000,   # Large buffer for stability
            "epsilon_start" : 1.0,
            "epsilon_end"   : 0.05,
            "epsilon_decay" : 200_000,   # Long exploration for 18 actions
            "target_update_interval": 10_000,
            "learning_starts": 50_000,   # Accumulate good experiences first
        },
        "total_timesteps": 300_000,
        "notes": "Optimized baseline for Tennis (18 actions) - full learning.",
    },
    {
        "name": "Exp2_LowerLR",
        "hyperparams": {
            "learning_rate" : 1e-4,
            "gamma"         : 0.99,
            "batch_size"    : 32,
            "buffer_size"   : 500_000,
            "epsilon_start" : 1.0,
            "epsilon_end"   : 0.05,
            "epsilon_decay" : 200_000,
            "target_update_interval": 10_000,
            "learning_starts": 50_000,
        },
        "total_timesteps": 300_000,
        "notes": "Lower learning rate - conservative, stable updates.",
    },
    {
        "name": "Exp3_HigherLR",
        "hyperparams": {
            "learning_rate" : 1e-3,
            "gamma"         : 0.99,
            "batch_size"    : 32,
            "buffer_size"   : 500_000,
            "epsilon_start" : 1.0,
            "epsilon_end"   : 0.05,
            "epsilon_decay" : 200_000,
            "target_update_interval": 10_000,
            "learning_starts": 50_000,
        },
        "total_timesteps": 300_000,
        "notes": "Higher learning rate - faster convergence risk/reward.",
    },
    {
        "name": "Exp4_LargeBatch",
        "hyperparams": {
            "learning_rate" : 5e-4,
            "gamma"         : 0.99,
            "batch_size"    : 64,
            "buffer_size"   : 500_000,
            "epsilon_start" : 1.0,
            "epsilon_end"   : 0.05,
            "epsilon_decay" : 200_000,
            "target_update_interval": 10_000,
            "learning_starts": 50_000,
        },
        "total_timesteps": 300_000,
        "notes": "Larger batch (64) - gradient stability, slower updates.",
    },
    {
        "name": "Exp5_SmallBatch",
        "hyperparams": {
            "learning_rate" : 5e-4,
            "gamma"         : 0.99,
            "batch_size"    : 16,
            "buffer_size"   : 500_000,
            "epsilon_start" : 1.0,
            "epsilon_end"   : 0.05,
            "epsilon_decay" : 200_000,
            "target_update_interval": 10_000,
            "learning_starts": 50_000,
        },
        "total_timesteps": 300_000,
        "notes": "Smaller batch (16) - faster individual updates, noisier.",
    },
    {
        "name": "Exp6_HighGamma",
        "hyperparams": {
            "learning_rate" : 5e-4,
            "gamma"         : 0.999,
            "batch_size"    : 32,
            "buffer_size"   : 500_000,
            "epsilon_start" : 1.0,
            "epsilon_end"   : 0.05,
            "epsilon_decay" : 200_000,
            "target_update_interval": 10_000,
            "learning_starts": 50_000,
        },
        "total_timesteps": 300_000,
        "notes": "High gamma (0.999) - strong long-term reward preference.",
    },
    {
        "name": "Exp7_LowGamma",
        "hyperparams": {
            "learning_rate" : 5e-4,
            "gamma"         : 0.95,
            "batch_size"    : 32,
            "buffer_size"   : 500_000,
            "epsilon_start" : 1.0,
            "epsilon_end"   : 0.05,
            "epsilon_decay" : 200_000,
            "target_update_interval": 10_000,
            "learning_starts": 50_000,
        },
        "total_timesteps": 300_000,
        "notes": "Low gamma (0.95) - prioritizes immediate rewards.",
    },
    {
        "name": "Exp8_FastExploration",
        "hyperparams": {
            "learning_rate" : 5e-4,
            "gamma"         : 0.99,
            "batch_size"    : 32,
            "buffer_size"   : 500_000,
            "epsilon_start" : 1.0,
            "epsilon_end"   : 0.05,
            "epsilon_decay" : 100_000,   # Fast epsilon decay
            "target_update_interval": 10_000,
            "learning_starts": 50_000,
        },
        "total_timesteps": 300_000,
        "notes": "Fast exploration decay - quicker shift to exploitation.",
    },
    {
        "name": "Exp9_SlowExploration",
        "hyperparams": {
            "learning_rate" : 5e-4,
            "gamma"         : 0.99,
            "batch_size"    : 32,
            "buffer_size"   : 500_000,
            "epsilon_start" : 1.0,
            "epsilon_end"   : 0.05,
            "epsilon_decay" : 300_000,   # Slow epsilon decay (extended beyond total_timesteps)
            "target_update_interval": 10_000,
            "learning_starts": 50_000,
        },
        "total_timesteps": 300_000,
        "notes": "Slow exploration decay - extended exploration phase.",
    },
    {
        "name": "Exp10_ExtendedTraining",
        "hyperparams": {
            "learning_rate" : 5e-4,
            "gamma"         : 0.99,
            "batch_size"    : 32,
            "buffer_size"   : 500_000,
            "epsilon_start" : 1.0,
            "epsilon_end"   : 0.05,
            "epsilon_decay" : 300_000,
            "target_update_interval": 10_000,
            "learning_starts": 50_000,
        },
        "total_timesteps": 500_000,    # 5x longer for deep convergence
        "notes": "Extended training (500k steps) for maximum learning on Tennis.",
    },
]

# ============================================================================
# SUMMARY HELPERS
# ============================================================================

def print_summary(results: List[Dict]) -> None:
    print(f"\n{'='*80}")
    print(f"  YOUR EXPERIMENTS — SUMMARY  ({len(results)} runs | MlpPolicy | Tennis-v5)")
    print(f"{'='*80}")
    header = f"  {'Experiment':<35} {'Mean Reward':>12} {'Max Reward':>11} {'Episodes':>9} {'Time(s)':>8}"
    print(header)
    print(f"  {'-'*75}")
    for r in results:
        if "error" in r:
            print(f"  {r['experiment']:<35}  ERROR: {r['error']}")
        else:
            print(
                f"  {r['experiment']:<35}"
                f"  {r['mean_reward']:>10.2f}"
                f"  {r['max_reward']:>10.2f}"
                f"  {r['episodes']:>9}"
                f"  {r['training_time']:>8.1f}s"
            )
    print(f"{'='*80}\n")


def save_master_results(results: List[Dict]) -> Path:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = BASE_DIR / f"results_custom_{ts}.json"
    clean = [{k: v for k, v in r.items() if k != "episode_rewards"} for r in results]
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"  ✓ Master results → {path}")
    return path


def print_hyperparameter_table(experiments: List[Dict]) -> None:
    """Print the assignment-style hyperparameter table for copy-pasting."""
    print(f"\n{'='*80}")
    print("  HYPERPARAMETER TABLE  (Your Experiments)")
    print(f"{'='*80}")
    print(f"  {'#':<4} {'Experiment':<35} {'lr':<8} {'gamma':<7} {'batch':<7} "
          f"{'ε_start':<9} {'ε_end':<8} {'ε_decay':<9}")
    print(f"  {'-'*78}")
    for i, exp in enumerate(experiments, 1):
        hp = exp["hyperparams"]
        print(
            f"  {i:<4} {exp['name']:<35}"
            f" {hp['learning_rate']:<8}"
            f" {hp['gamma']:<7}"
            f" {hp['batch_size']:<7}"
            f" {hp.get('epsilon_start', 1.0):<9}"
            f" {hp['epsilon_end']:<8}"
            f" {hp.get('epsilon_decay', 50000):<9}"
        )
    print(f"{'='*80}\n")


# ============================================================================
# MAIN RUNNER
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train DQN agent on Tennis-v5 - Run experiments one at a time for full learning"
    )
    parser.add_argument(
        "--exp",
        type=int,
        default=None,
        help="Run specific experiment number (1-10). If not provided, runs all 10 experiments sequentially.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available experiments and exit."
    )
    
    args = parser.parse_args()

    if args.list:
        print_hyperparameter_table(YOUR_EXPERIMENTS)
        for i, exp in enumerate(YOUR_EXPERIMENTS, 1):
            print(f"{i}. {exp['name']}: {exp['notes']}")
        sys.exit(0)

    print("\n" + "="*70)
    print("  Deep Q-Network  |  MlpPolicy  |  Your 10 Experiments")
    print(f"  Environment : {ENV_NAME}")
    print(f"  Action Space: 18 actions (see Tennis documentation)")
    print(f"  Output dir  : {BASE_DIR}")
    print("="*70)

    print_hyperparameter_table(YOUR_EXPERIMENTS)

    results: List[Dict] = []

    if args.exp:
        # Run single experiment
        if args.exp < 1 or args.exp > len(YOUR_EXPERIMENTS):
            print(f"✗ Invalid experiment number. Choose 1-{len(YOUR_EXPERIMENTS)}")
            sys.exit(1)
        
        exp_idx = args.exp - 1
        exp = YOUR_EXPERIMENTS[exp_idx]
        print(f"\n>>> Running SINGLE experiment {args.exp}/{len(YOUR_EXPERIMENTS)}: {exp['name']}")
        print(f"    Notes: {exp['notes']}")
        print(f"    Timesteps: {exp.get('total_timesteps', TOTAL_TIMESTEPS_DEFAULT):,}")
        
        try:
            result = train_experiment(
                exp_name         = exp["name"],
                hyperparams      = exp["hyperparams"],
                total_timesteps  = exp.get("total_timesteps", TOTAL_TIMESTEPS_DEFAULT),
            )
            result["notes"] = exp["notes"]
            results.append(result)
            print_summary(results)
        except Exception as e:
            print(f"\n  ✗ Experiment {exp['name']} failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Run all experiments sequentially
        print(f"\n>>> Running ALL {len(YOUR_EXPERIMENTS)} experiments sequentially (one at a time)")
        print("    Each experiment will complete fully before the next starts.\n")
        
        for i, exp in enumerate(YOUR_EXPERIMENTS, 1):
            print(f"\n>>> Starting experiment {i}/{len(YOUR_EXPERIMENTS)}: {exp['name']}")
            print(f"    Notes: {exp['notes']}")
            print(f"    Timesteps: {exp.get('total_timesteps', TOTAL_TIMESTEPS_DEFAULT):,}")
            try:
                result = train_experiment(
                    exp_name         = exp["name"],
                    hyperparams      = exp["hyperparams"],
                    total_timesteps  = exp.get("total_timesteps", TOTAL_TIMESTEPS_DEFAULT),
                )
                result["notes"] = exp["notes"]
                results.append(result)
            except Exception as e:
                print(f"\n  ✗ Experiment {exp['name']} failed: {e}")
                results.append({"experiment": exp["name"], "policy": POLICY_TYPE, "error": str(e)})

        print_summary(results)

    save_master_results(results)
    print("  ✓ Experiment(s) complete.\n")


if __name__ == "__main__":
    main()
