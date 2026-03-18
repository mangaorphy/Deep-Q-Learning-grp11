"""
Deep Q-Network (DQN) Training Script for Atari Tennis
======================================================
A clean, modular, and production-quality training script for DQN agents
using Stable Baselines3 and Gymnasium.

This script is designed for easy experimentation with multiple hyperparameter
configurations. Each group member can easily run 10 different experiments.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np

# Try to import and register ALE
try:
    import ale_py
    gym.register_envs(ale_py)
    print("✓ ALE environments registered successfully")
except Exception as e:
    print(f"Warning: ALE registration issue: {e}")

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Try to import matplotlib for optional plotting
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting disabled.")


# ============================================================================
# CONFIGURATION SECTION - MODIFY HERE FOR EXPERIMENTATION
# ============================================================================

# Learning and environment parameters
CONFIG = {
    # General settings
    "env_name": "ALE/Tennis-v5",
    "policy_type": "CnnPolicy",  # Options: "MlpPolicy" or "CnnPolicy"
    "total_timesteps": 100000,
    "seed": 42,
    
    # DQN Hyperparameters (CRITICAL: These are easily configurable for experiments)
    "learning_rate": 1e-4,
    "gamma": 0.99,          # Discount factor
    "batch_size": 32,
    "buffer_size": 100000,
    "epsilon_start": 1.0,   # Initial exploration rate
    "epsilon_end": 0.05,    # Final exploration rate
    "epsilon_decay": 50000, # Timesteps to decay epsilon
    "target_update_interval": 1000,
    "learning_starts": 1000,
    
    # Output settings
    "model_save_path": "dqn_model.zip",
    "log_dir": "logs",
    "create_experiment_log": True,
}


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def make_env(env_name: str, seed: int = 42) -> gym.Env:
    """
    Create and configure the Atari Tennis environment with preprocessing wrappers.
    
    Args:
        env_name: Name of the environment (e.g., "ALE/Tennis-v5")
        seed: Random seed for reproducibility
    
    Returns:
        Configured gymnasium environment with preprocessing wrappers
    """
    env = gym.make(env_name, render_mode=None, frameskip=1)
    env.reset(seed=seed)
    
    # Add Monitor wrapper for automatic episode tracking
    env = Monitor(env, info_keywords=("lives",))
    
    # Apply Atari preprocessing wrappers from gymnasium
    # These wrappers handle:
    # - Grayscale conversion
    # - Frame resizing to 84x84
    # - Frame stacking (4 frames) - important for CNNPolicy
    # Note: frame_skip is NOT applied here since we set frameskip=1 above
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=1,  # Already handled in gym.make() call above
        screen_size=84,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        grayscale_newaxis=False,  # Don't add extra dimension for grayscale
        scale_obs=True,
    )
    
    # Stack frames for temporal information - creates (4, 84, 84) shape
    env = gym.wrappers.FrameStack(env, num_stack=4)
    
    return env


def validate_environment(env: gym.Env, policy_type: str) -> None:
    """
    Validate environment configuration and compatibility with policy type.
    
    Args:
        env: The gymnasium environment to validate
        policy_type: Policy type ("MlpPolicy" or "CnnPolicy")
    
    Raises:
        ValueError: If environment is incompatible with policy type
    """
    obs_space = env.observation_space
    action_space = env.action_space
    
    print(f"\n{'='*70}")
    print(f"Environment Configuration")
    print(f"{'='*70}")
    print(f"Observation Space: {obs_space}")
    print(f"Action Space: {action_space}")
    
    # CNNPolicy requires 3 or 4D observation space (batch, height, width, channels)
    if policy_type == "CnnPolicy":
        if len(obs_space.shape) < 3:
            raise ValueError(
                f"CNNPolicy requires at least 3D observation space. "
                f"Got: {obs_space.shape}"
            )
    
    print(f"Policy Type: {policy_type}")
    print(f"✓ Environment is compatible with {policy_type}")
    print(f"{'='*70}\n")


# ============================================================================
# TRAINING
# ============================================================================

class RewardCallback(BaseCallback):
    """
    Custom callback to track training progress and log rewards.
    """
    
    def __init__(self, log_freq: int = 1000):
        super().__init__()
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        """Called after every environment step."""
        # Try to get episode rewards from the environment
        # Different environment types may track this differently
        try:
            # Check if Monitor wrapper is available
            if hasattr(self.model.env, 'get_attr'):
                # For vectorized environments
                info = self.model.env.get_attr("episode")
                if info and info[0] != self.episode_count:
                    self.episode_count = info[0]
                    if hasattr(self.model.env, 'return_queue') and len(self.model.env.return_queue) > 0:
                        ep_reward, ep_length = self.model.env.return_queue[-1]
                        self.episode_rewards.append(ep_reward)
                        self.episode_lengths.append(ep_length)
                        
                        if len(self.episode_rewards) % self.log_freq == 0:
                            mean_reward = np.mean(self.episode_rewards[-100:])
                            print(
                                f"Episodes: {self.episode_count} | "
                                f"Timesteps: {self.num_timesteps} | "
                                f"Mean Reward (last 100): {mean_reward:.2f}"
                            )
        except (AttributeError, IndexError):
            # Fallback: silently ignore if episode tracking not available
            pass
        
        return True


def train_model(
    env: gym.Env,
    policy_type: str,
    total_timesteps: int,
    hyperparameters: Dict,
    model_save_path: str = "dqn_model.zip",
) -> Tuple[DQN, Dict]:
    """
    Train a DQN agent on the provided environment.
    
    Args:
        env: Gymnasium environment
        policy_type: "MlpPolicy" or "CnnPolicy"
        total_timesteps: Total number of environment steps to train
        hyperparameters: Dictionary of DQN hyperparameters
        model_save_path: Path to save the trained model
    
    Returns:
        Tuple of (trained model, training statistics)
    """
    print(f"\n{'='*70}")
    print(f"Training Configuration")
    print(f"{'='*70}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Policy Type: {policy_type}")
    
    print(f"\nHyperparameters:")
    for key, value in hyperparameters.items():
        print(f"  {key}: {value}")
    print(f"{'='*70}\n")
    
    # Create DQN model with specified hyperparameters
    model = DQN(
        policy_type,
        env,
        learning_rate=hyperparameters["learning_rate"],
        gamma=hyperparameters["gamma"],
        batch_size=hyperparameters["batch_size"],
        buffer_size=hyperparameters["buffer_size"],
        target_update_interval=hyperparameters["target_update_interval"],
        learning_starts=hyperparameters["learning_starts"],
        exploration_fraction=hyperparameters["epsilon_decay"] / total_timesteps,
        exploration_initial_eps=hyperparameters["epsilon_start"],
        exploration_final_eps=hyperparameters["epsilon_end"],
        verbose=1,  # Set to 1 for detailed output
        device="auto",
        policy_kwargs={"normalize_images": False},  # Don't normalize channel-first images
    )
    
    # Train the model
    print("Starting training...")
    start_time = datetime.now()
    
    callback = RewardCallback(log_freq=1000)
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    # Save the trained model
    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    model.save(model_save_path)
    print(f"\n✓ Model saved to: {model_save_path}")
    
    # Compile training statistics
    stats = {
        "total_timesteps": total_timesteps,
        "training_time_seconds": training_time,
        "episodes_completed": callback.episode_count,
        "mean_episode_reward": float(np.mean(callback.episode_rewards)) if callback.episode_rewards else 0,
        "mean_episode_length": float(np.mean(callback.episode_lengths)) if callback.episode_lengths else 0,
        "max_episode_reward": float(np.max(callback.episode_rewards)) if callback.episode_rewards else 0,
        "episode_rewards": callback.episode_rewards,
        "episode_lengths": callback.episode_lengths,
    }
    
    return model, stats


def print_training_summary(stats: Dict, hyperparameters: Dict) -> None:
    """
    Print a formatted summary of training results.
    
    Args:
        stats: Training statistics dictionary
        hyperparameters: Hyperparameters used for training
    """
    print(f"\n{'='*70}")
    print(f"Training Summary")
    print(f"{'='*70}")
    print(f"Total Timesteps: {stats['total_timesteps']:,}")
    print(f"Training Time: {stats['training_time_seconds']:.2f} seconds")
    print(f"Episodes Completed: {stats['episodes_completed']}")
    print(f"\nPerformance Metrics:")
    print(f"  Mean Episode Reward: {stats['mean_episode_reward']:.2f}")
    print(f"  Max Episode Reward: {stats['max_episode_reward']:.2f}")
    print(f"  Mean Episode Length: {stats['mean_episode_length']:.2f}")
    print(f"\nHyperparameters Used:")
    for key, value in hyperparameters.items():
        if key not in ["buffer_size", "target_update_interval", "learning_starts"]:
            print(f"  {key}: {value}")
    print(f"{'='*70}\n")


# ============================================================================
# EXPERIMENT LOGGING
# ============================================================================

def save_experiment_log(
    stats: Dict,
    hyperparameters: Dict,
    config: Dict,
    log_dir: str = "logs",
) -> str:
    """
    Save experiment results to a JSON file for later analysis.
    
    Args:
        stats: Training statistics
        hyperparameters: Hyperparameters used
        config: Full configuration
        log_dir: Directory to save logs
    
    Returns:
        Path to saved log file
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create experiment record (exclude long arrays)
    experiment_record = {
        "timestamp": datetime.now().isoformat(),
        "config": {k: v for k, v in config.items() if k != "episode_rewards"},
        "hyperparameters": hyperparameters,
        "statistics": {k: v for k, v in stats.items() if k not in ["episode_rewards", "episode_lengths"]},
    }
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{log_dir}/experiment_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(experiment_record, f, indent=2)
    
    print(f"✓ Experiment log saved to: {filename}")
    return filename


# ============================================================================
# PLOTTING (BONUS)
# ============================================================================

def plot_training_results(stats: Dict, save_path: str = "training_results.png") -> None:
    """
    Plot training results (episode rewards and lengths).
    
    Args:
        stats: Training statistics containing episode_rewards and episode_lengths
        save_path: Path to save the plot
    """
    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib not available. Skipping plot generation.")
        return
    
    episode_rewards = stats.get("episode_rewards", [])
    episode_lengths = stats.get("episode_lengths", [])
    
    if not episode_rewards:
        print("No episode data available for plotting.")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot episode rewards
    axes[0].plot(episode_rewards, alpha=0.6, label="Episode Reward")
    axes[0].plot(
        np.convolve(episode_rewards, np.ones(100) / 100, mode="valid"),
        label="100-episode Moving Average",
        linewidth=2,
    )
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Training Progress: Episode Rewards")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot episode lengths
    axes[1].plot(episode_lengths, alpha=0.6, label="Episode Length", color="orange")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].set_title("Training Progress: Episode Lengths")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ Training plot saved to: {save_path}")


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    """
    Main training script. Modify CONFIG at the top to change hyperparameters.
    """
    print("\n" + "="*70)
    print("Deep Q-Network (DQN) Training for Atari Tennis")
    print("="*70)
    
    # Set random seeds for reproducibility
    np.random.seed(CONFIG["seed"])
    
    # Create environment
    print("\nCreating environment...")
    env = make_env(CONFIG["env_name"], seed=CONFIG["seed"])
    
    # Validate environment
    validate_environment(env, CONFIG["policy_type"])
    
    # Extract hyperparameters
    hyperparameters = {
        "learning_rate": CONFIG["learning_rate"],
        "gamma": CONFIG["gamma"],
        "batch_size": CONFIG["batch_size"],
        "buffer_size": CONFIG["buffer_size"],
        "epsilon_start": CONFIG["epsilon_start"],
        "epsilon_end": CONFIG["epsilon_end"],
        "epsilon_decay": CONFIG["epsilon_decay"],
        "target_update_interval": CONFIG["target_update_interval"],
        "learning_starts": CONFIG["learning_starts"],
    }
    
    # Train model
    model, stats = train_model(
        env=env,
        policy_type=CONFIG["policy_type"],
        total_timesteps=CONFIG["total_timesteps"],
        hyperparameters=hyperparameters,
        model_save_path=CONFIG["model_save_path"],
    )
    
    # Print summary
    print_training_summary(stats, hyperparameters)
    
    # Save experiment log
    if CONFIG["create_experiment_log"]:
        save_experiment_log(stats, hyperparameters, CONFIG, CONFIG["log_dir"])
    
    # Plot results (bonus)
    plot_training_results(stats, save_path="training_results.png")
    
    # Close environment
    env.close()
    
    print("✓ Training completed successfully!")
    return model, stats


# ============================================================================
# 10 EXPERIMENTS - MEMBER 3
# ============================================================================

EXPERIMENTS = [
    {
        "name": "Exp1_AggressiveLR",
        "config_changes": {"learning_rate": 0.001, "gamma": 0.95, "batch_size": 64},
    },
    {
        "name": "Exp2_ConservativeLR",
        "config_changes": {"learning_rate": 5e-6, "gamma": 0.999, "batch_size": 16},
    },
    {
        "name": "Exp3_BalancedConfig",
        "config_changes": {"learning_rate": 1.5e-4, "gamma": 0.98, "batch_size": 32},
    },
    {
        "name": "Exp4_HighExplorationHighLR",
        "config_changes": {"learning_rate": 5e-4, "gamma": 0.99, "batch_size": 32, "epsilon_end": 0.15},
    },
    {
        "name": "Exp5_LowExplorationLowLR",
        "config_changes": {"learning_rate": 5e-5, "gamma": 0.99, "batch_size": 32, "epsilon_end": 0.01},
    },
    {
        "name": "Exp6_LargeBatchSmallLR",
        "config_changes": {"learning_rate": 5e-5, "gamma": 0.99, "batch_size": 128},
    },
    {
        "name": "Exp7_SmallBatchLargeLR",
        "config_changes": {"learning_rate": 5e-4, "gamma": 0.99, "batch_size": 16},
    },
    {
        "name": "Exp8_VeryHighGamma",
        "config_changes": {"learning_rate": 1e-4, "gamma": 0.9999, "batch_size": 32},
    },
    {
        "name": "Exp9_VeryLowGamma",
        "config_changes": {"learning_rate": 1e-4, "gamma": 0.9, "batch_size": 32},
    },
    {
        "name": "Exp10_ExtendedTraining",
        "config_changes": {"learning_rate": 1e-4, "gamma": 0.99, "batch_size": 32, "total_timesteps": 300000},
    },
]


def run_all_experiments():
    """Run all 10 experiments with both CnnPolicy and MlpPolicy."""
    import json
    os.makedirs("models", exist_ok=True)
    
    results = []
    policy_comparison = []
    policies = ["CnnPolicy", "MlpPolicy"]
    total_experiments = len(EXPERIMENTS) * len(policies)
    current = 0
    
    for exp in EXPERIMENTS:
        exp_comparison = {
            "experiment": exp["name"],
            "config": exp["config_changes"],
            "cnn_results": {},
            "mlp_results": {},
        }
        
        for policy in policies:
            current += 1
            print(f"\n{'='*70}")
            print(f"Running Experiment {current}/{total_experiments}")
            print(f"Experiment: {exp['name']}")
            print(f"Policy: {policy}")
            print(f"{'='*70}\n")
            
            # Save original config
            original_config = CONFIG.copy()
            
            # Update config with experiment changes
            CONFIG.update(exp["config_changes"])
            CONFIG["policy_type"] = policy
            
            # Set model path
            base_name = exp["name"].lower().replace(" ", "_")
            CONFIG["model_save_path"] = f"models/{base_name}_{policy.lower()}.zip"
            
            try:
                # Train
                model, stats = main()
                
                # Store results
                result = {
                    "experiment": exp["name"],
                    "policy": policy,
                    "config": exp["config_changes"],
                    "mean_reward": stats["mean_episode_reward"],
                    "max_reward": stats["max_episode_reward"],
                    "episodes": stats["episodes_completed"],
                    "time": stats["training_time_seconds"],
                }
                results.append(result)
                
                # Store for comparison
                if policy == "CnnPolicy":
                    exp_comparison["cnn_results"] = stats
                else:
                    exp_comparison["mlp_results"] = stats
                    
            except Exception as e:
                print(f"\n✗ Failed: {e}")
                results.append({
                    "experiment": exp["name"],
                    "policy": policy,
                    "error": str(e)
                })
            finally:
                # Restore config
                CONFIG.clear()
                CONFIG.update(original_config)
        
        # Determine winner
        if exp_comparison["cnn_results"] and exp_comparison["mlp_results"]:
            cnn_reward = exp_comparison["cnn_results"].get("mean_episode_reward", 0)
            mlp_reward = exp_comparison["mlp_results"].get("mean_episode_reward", 0)
            exp_comparison["winner"] = "CnnPolicy" if cnn_reward > mlp_reward else "MlpPolicy"
            exp_comparison["reward_diff"] = abs(cnn_reward - mlp_reward)
        
        policy_comparison.append(exp_comparison)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}\n")
    
    cnn_wins = sum(1 for c in policy_comparison if c.get("winner") == "CnnPolicy")
    mlp_wins = sum(1 for c in policy_comparison if c.get("winner") == "MlpPolicy")
    
    print(f"CnnPolicy wins: {cnn_wins}/10")
    print(f"MlpPolicy wins: {mlp_wins}/10")
    print(f"\nResults saved to: results_{timestamp}.json\n")


if __name__ == "__main__":
    run_all_experiments()
