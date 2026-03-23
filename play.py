"""
play.py — ALE/Tennis-v5  |  Student: Sage  |  Policy: CnnPolicy
────────────────────────────────────────────────────────────────
Evaluate a trained DQN model and print a clean results report.
Loads any .zip model saved by train.py and runs N evaluation episodes.

Usage:
    python3 play.py                                    # plays dqn_model.zip (best)
    python3 play.py --model models/exp1_cnn_baseline_cnnpolicy.zip
    python3 play.py --model dqn_model.zip --episodes 5
    python3 play.py --all-models                       # evaluate every .zip in models/
    python3 play.py --compare                          # compare all experiments vs best
"""

import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

gym.register_envs(ale_py)

SAGE_DIR   = os.path.dirname(os.path.abspath(__file__))
ENV_ID     = "ALE/Tennis-v5"
BEST_MODEL = os.path.join(SAGE_DIR, "dqn_model.zip")
MODELS_DIR = os.path.join(SAGE_DIR, "models")

# Other student directories are evaluated dynamically below.


def make_eval_env(seed: int = 0) -> VecFrameStack:
    """Single eval env — same preprocessing as training."""
    env = make_atari_env(ENV_ID, n_envs=1, vec_env_cls=DummyVecEnv, seed=seed)
    return VecFrameStack(env, n_stack=4)


def evaluate_model(model_path: str, n_episodes: int = 5,
                   deterministic: bool = True) -> dict:
    """
    Run n_episodes with the loaded model.
    Returns dict with mean/max/min/std reward and per-episode list.
    """
    if not os.path.exists(model_path):
        return {"error": f"Model not found: {model_path}"}

    env   = make_eval_env(seed=0)
    model = DQN.load(model_path, env=env)

    rewards, lengths = [], []
    obs = env.reset()
    ep_reward, ep_length = 0.0, 0
    ep_done = 0

    t0 = time.time()
    while ep_done < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)
        ep_reward += float(reward[0])
        ep_length += 1
        if done[0]:
            rewards.append(ep_reward)
            lengths.append(ep_length)
            ep_done += 1
            ep_reward, ep_length = 0.0, 0
            obs = env.reset()

    elapsed = time.time() - t0
    env.close()

    return {
        "model":       model_path,
        "episodes":    n_episodes,
        "mean_reward": round(float(np.mean(rewards)), 2),
        "max_reward":  round(float(np.max(rewards)),  2),
        "min_reward":  round(float(np.min(rewards)),  2),
        "std_reward":  round(float(np.std(rewards)),  2),
        "mean_length": round(float(np.mean(lengths)), 1),
        "all_rewards": rewards,
        "time_s":      round(elapsed, 1),
    }


def print_eval_result(res: dict, label: str = ""):
    if "error" in res:
        print(f"  ERROR: {res['error']}")
        return
    name = label or os.path.basename(res["model"])
    print(f"\n  {name}")
    print(f"  {'─' * 50}")
    print(f"  Mean reward : {res['mean_reward']:>8.2f}")
    print(f"  Max  reward : {res['max_reward']:>8.2f}")
    print(f"  Min  reward : {res['min_reward']:>8.2f}")
    print(f"  Std  reward : {res['std_reward']:>8.2f}")
    print(f"  Mean length : {res['mean_length']:>8.1f} steps/episode")
    print(f"  Episodes    : {res['episodes']}")
    print(f"  Eval time   : {res['time_s']}s")
    if res["all_rewards"]:
        ep_strs = [f"{r:>7.1f}" for r in res["all_rewards"]]
        print(f"  Per-episode : {' '.join(ep_strs)}")


def save_comparison_chart(results: list, out_path: str):
    """Bar chart comparing mean rewards across evaluated models."""
    labels = [os.path.basename(r["model"]).replace("_cnnpolicy.zip", "")
              .replace("sage_", "") for r in results]
    means  = [r["mean_reward"] for r in results]
    maxes  = [r["max_reward"]  for r in results]

    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0d1f2d")
    ax.set_facecolor("#091520")
    ax.tick_params(colors="#7ec8e3", labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a4a6b")

    x, w = np.arange(len(labels)), 0.38
    ax.bar(x - w / 2, means, width=w, color="#00c9ff", alpha=0.88, label="Mean reward")
    ax.bar(x + w / 2, maxes, width=w, color="#6c5ce7", alpha=0.88, label="Max reward")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#b2ebf2", fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("Reward", color="#7ec8e3")
    ax.set_title("Sage's Model Comparison  ·  CnnPolicy  ·  Tennis-v5",
                 color="#b2ebf2", fontsize=12, pad=10)
    ax.legend(fontsize=9, facecolor="#0d1f2d", labelcolor="#b2ebf2")

    best_idx = int(np.argmax(means))
    ax.get_xticklabels()[best_idx].set_color("#00e5ff")
    ax.get_xticklabels()[best_idx].set_fontweight("bold")

    fig.tight_layout(pad=1.5)
    fig.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  Comparison chart → {out_path}")


def main():
    p = argparse.ArgumentParser(
        description="Evaluate trained CnnPolicy DQN on ALE/Tennis-v5  —  Sage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",      default=BEST_MODEL,
                   help="Path to .zip model file")
    p.add_argument("--episodes",   type=int, default=5,
                   help="Number of evaluation episodes")
    p.add_argument("--stochastic", action="store_true",
                   help="Use stochastic policy (default: deterministic greedy)")
    p.add_argument("--all-models", action="store_true",
                   help="Evaluate every .zip in models/ directory")
    p.add_argument("--compare",    action="store_true",
                   help="Evaluate all models and save a comparison chart")
    args = p.parse_args()

    deterministic = not args.stochastic

    print("\n" + "=" * 60)
    print("  SAGE  |  DQN Evaluation  |  CnnPolicy  |  Tennis-v5")
    print("=" * 60)
    print(f"  Policy mode : {'deterministic' if deterministic else 'stochastic'}")
    print(f"  Episodes    : {args.episodes}")
    print("=" * 60)

    if args.all_models or args.compare:
        if not os.path.isdir(MODELS_DIR):
            print(f"\n  No models directory found at: {MODELS_DIR}")
            print("  Run train.py --all first.\n")
            return

        model_files = []
        if os.path.isdir(MODELS_DIR):
            model_files.extend(sorted([
                os.path.join(MODELS_DIR, f)
                for f in os.listdir(MODELS_DIR)
                if f.endswith(".zip")
            ]))

        for student in ["kariza", "orpheus", "Emmanuel"]:
            student_dir = os.path.join(SAGE_DIR, student)
            student_models_dir = os.path.join(student_dir, "models")
            
            best = os.path.join(student_dir, "dqn_best.zip")
            if os.path.exists(best):
                model_files.append(best)

            latest = os.path.join(student_dir, "dqn_latest.zip")
            if os.path.exists(latest):
                model_files.append(latest)

            if os.path.isdir(student_models_dir):
                model_files.extend(sorted([
                    os.path.join(student_models_dir, f)
                    for f in os.listdir(student_models_dir)
                    if f.endswith(".zip")
                ]))

        if not model_files:
            print(f"\n  No .zip models found in {MODELS_DIR}")
            return

        print(f"\n  Found {len(model_files)} model(s) to evaluate...\n")
        all_results = []
        for mf in model_files:
            print(f"  Evaluating: {os.path.basename(mf)}")
            res = evaluate_model(mf, n_episodes=args.episodes,
                                 deterministic=deterministic)
            all_results.append(res)
            print_eval_result(res)

        if args.compare and all_results:
            chart_path = os.path.join(SAGE_DIR, "comparison_chart.png")
            save_comparison_chart(all_results, chart_path)

        # Summary table
        print(f"\n{'─'*70}")
        print(f"  {'Model':<40} {'Mean':>7} {'Max':>7} {'Std':>7}")
        print(f"  {'─'*40} {'─'*7} {'─'*7} {'─'*7}")
        for res in sorted(all_results, key=lambda x: x.get("mean_reward", -999),
                           reverse=True):
            if "error" not in res:
                label = os.path.basename(res["model"])[:40]
                print(f"  {label:<40} {res['mean_reward']:>7.2f} "
                      f"{res['max_reward']:>7.2f} {res['std_reward']:>7.2f}")
        print(f"{'─'*70}\n")

        # Save JSON
        out_json = os.path.join(SAGE_DIR, "eval_results.json")
        with open(out_json, "w") as fh:
            json.dump([r for r in all_results if "error" not in r], fh, indent=2)
        print(f"  Eval results → {out_json}")

    else:
        res = evaluate_model(args.model, n_episodes=args.episodes,
                             deterministic=deterministic)
        print_eval_result(res, label=os.path.basename(args.model))

    print()


if __name__ == "__main__":
    main()
