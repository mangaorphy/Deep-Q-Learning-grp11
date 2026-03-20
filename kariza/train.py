"""
train.py — ALE/Tennis-v5  |  Student: Kariza  |  Policy: CnnPolicy
────────────────────────────────────────────────────────────────────
Train DQN agent on Atari Tennis using CnnPolicy.
Vectorized with make_atari_env + VecFrameStack for speed.
Saves checkpoint every --save-freq steps so play.py can watch live.
Appends results to results_YYYYMMDD_HHMMSS.json after each run.

HOW TO ADD YOUR OWN EXPERIMENTS:
  Find the EXPERIMENTS list near the bottom of this file (~line 400).
  Add a new dict with "name" and "config_changes" keys.
  Run: python3 train.py

Usage:
    python3 train.py                          # runs single experiment (Baseline)
    python3 train.py --exp Baseline           # run specific experiment by name
    python3 train.py --all                    # run all experiments in sequence
    python3 train.py --lr 0.0001 --gamma 0.99 --batch 32 --timesteps 500000
"""

import argparse
import csv
import json
import os
import platform
import time
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, DummyVecEnv

gym.register_envs(ale_py)

# ── Constants ─────────────────────────────────────────────────────────────────
ENV_ID      = "ALE/Tennis-v5"
CHECKPOINT  = "dqn_latest.zip"
BEST_MODEL  = "dqn_best.zip"
BEST_SCORE_F = "best_score.json"
MODELS_DIR  = "models"

ACTION_NAMES = [
    "NOOP","FIRE","UP","RIGHT","LEFT","DOWN",
    "UPRIGHT","UPLEFT","DOWNRIGHT","DOWNLEFT",
    "UPFIRE","RIGHTFIRE","LEFTFIRE","DOWNFIRE",
    "UPRIGHTFIRE","UPLEFTFIRE","DOWNRIGHTFIRE","DOWNLEFTFIRE",
]

# ── Default hyperparameters ───────────────────────────────────────────────────
DEFAULTS = {
    "policy":           "CnnPolicy",
    "learning_rate":    1e-4,
    "gamma":            0.99,
    "batch_size":       32,
    "buffer_size":      50_000,
    "epsilon_start":    1.0,
    "epsilon_end":      0.05,
    "epsilon_decay":    0.10,
    "total_timesteps":  100_000,
    "n_envs":           4,
    "learning_starts":  5_000,
    "target_update":    1_000,
    "train_freq":       4,
}

# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENTS LIST — add your experiments here
# Each dict needs: "name" and "config_changes"
# config_changes overrides any key in DEFAULTS above
# ──────────────────────────────────────────────────────────────────────────────
EXPERIMENTS = [
    {
        "name": "Exp1_Baseline",
        "config_changes": {
            "learning_rate": 1e-4,
            "gamma":         0.99,
            "batch_size":    32,
            "epsilon_end":   0.05,
            "total_timesteps": 100_000,
        },
    },
    {
        "name": "Exp2_AggressiveLR",
        "config_changes": {
            "learning_rate": 0.001,
            "gamma":         0.95,
            "batch_size":    64,
            "epsilon_end":   0.05,
            "total_timesteps": 100_000,
        },
    },
    {
        "name": "Exp3_LowLR",
        "config_changes": {
            "learning_rate": 1e-5,
            "gamma":         0.99,
            "batch_size":    32,
            "epsilon_end":   0.05,
            "total_timesteps": 100_000,
        },
    },
    {
        "name": "Exp4_LowGamma",
        "config_changes": {
            "learning_rate": 1e-4,
            "gamma":         0.90,
            "batch_size":    32,
            "epsilon_end":   0.05,
            "total_timesteps": 100_000,
        },
    },
    {
        "name": "Exp5_HighGamma",
        "config_changes": {
            "learning_rate": 1e-4,
            "gamma":         0.999,
            "batch_size":    32,
            "epsilon_end":   0.05,
            "total_timesteps": 100_000,
        },
    },
    {
        "name": "Exp6_LargeBatch",
        "config_changes": {
            "learning_rate": 1e-4,
            "gamma":         0.99,
            "batch_size":    128,
            "epsilon_end":   0.05,
            "total_timesteps": 100_000,
        },
    },
    {
        "name": "Exp7_SmallBatch",
        "config_changes": {
            "learning_rate": 1e-4,
            "gamma":         0.99,
            "batch_size":    16,
            "epsilon_end":   0.05,
            "total_timesteps": 100_000,
        },
    },
    {
        "name": "Exp8_HighEpsilonEnd",
        "config_changes": {
            "learning_rate": 1e-4,
            "gamma":         0.99,
            "batch_size":    32,
            "epsilon_end":   0.20,
            "total_timesteps": 100_000,
        },
    },
    {
        "name": "Exp9_SlowEpsilonDecay",
        "config_changes": {
            "learning_rate": 1e-4,
            "gamma":         0.99,
            "batch_size":    32,
            "epsilon_end":   0.05,
            "epsilon_decay": 0.50,
            "total_timesteps": 100_000,
        },
    },
    {
        "name": "Exp10_BestGuess",
        "config_changes": {
            "learning_rate": 5e-4,
            "gamma":         0.995,
            "batch_size":    64,
            "epsilon_end":   0.05,
            "epsilon_decay": 0.15,
            "total_timesteps": 100_000,
        },
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Best score tracker
# ──────────────────────────────────────────────────────────────────────────────
def _get_best_score() -> float:
    if os.path.exists(BEST_SCORE_F):
        with open(BEST_SCORE_F) as f:
            return json.load(f).get("mean_reward", -999)
    return -999.0


def _set_best_score(mean_reward: float, exp_name: str):
    with open(BEST_SCORE_F, "w") as f:
        json.dump({
            "mean_reward": mean_reward,
            "experiment":  exp_name,
            "saved_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Environment builder
# ──────────────────────────────────────────────────────────────────────────────
def make_env(n_envs: int) -> VecFrameStack:
    """
    Wraps Tennis-v5 with standard Atari preprocessing:
      NoopReset, MaxAndSkip, EpisodicLife, FireReset,
      ClipReward, WarpFrame(84x84 grayscale), VecFrameStack(4)
    Output obs shape: (n_envs, 4, 84, 84)
    CnnPolicy applies convolutional filters to detect ball, player, court positions.
    """
    vec_cls = DummyVecEnv
    if n_envs > 1 and platform.system() != "Darwin":
        vec_cls = SubprocVecEnv
    env = make_atari_env(ENV_ID, n_envs=n_envs,
                         vec_env_cls=vec_cls, seed=42)
    return VecFrameStack(env, n_stack=4)


# ──────────────────────────────────────────────────────────────────────────────
# Callback
# ──────────────────────────────────────────────────────────────────────────────
class TrainCallback(BaseCallback):
    def __init__(self, total_steps: int, log_path: str,
                 save_freq: int = 10_000, print_freq: int = 5):
        super().__init__()
        self.total_steps = total_steps
        self.log_path    = log_path
        self.save_freq   = save_freq
        self.print_freq  = print_freq
        self.ep_rewards  = []
        self.ep_lengths  = []
        self._ep_count   = 0
        self._last_save  = 0
        self._start      = time.time()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["episode","reward","length","elapsed_s","steps"])

    def _on_step(self) -> bool:
        # periodic checkpoint
        if self.num_timesteps - self._last_save >= self.save_freq:
            self.model.save(CHECKPOINT)
            self._last_save = self.num_timesteps

        for info in self.locals.get("infos", []):
            if "episode" in info:
                r = float(info["episode"]["r"])
                l = int(info["episode"]["l"])
                self._ep_count += 1
                self.ep_rewards.append(r)
                self.ep_lengths.append(l)
                with open(self.log_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        self._ep_count, r, l,
                        round(time.time()-self._start, 1),
                        self.num_timesteps,
                    ])
                if self._ep_count % self.print_freq == 0:
                    recent = self.ep_rewards[-self.print_freq:]
                    sps    = self.num_timesteps / max(time.time()-self._start, 1)
                    pct    = self.num_timesteps / self.total_steps * 100
                    print(f"  Ep {self._ep_count:>4} | "
                          f"Steps {self.num_timesteps:>8,} ({pct:.0f}%) | "
                          f"SPS {sps:>5.0f} | "
                          f"Reward mean={np.mean(recent):>7.2f}  "
                          f"best={max(self.ep_rewards):>6.2f}")
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Chart
# ──────────────────────────────────────────────────────────────────────────────
def save_chart(rewards, lengths, out_dir, name):
    if not rewards:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), facecolor="#2b1020")
    xs = list(range(1, len(rewards)+1))
    for ax, data, col, title in [
        (ax1, rewards, "#ff3860", "Episode Reward"),
        (ax2, lengths, "#8b5cf6", "Episode Length"),
    ]:
        ax.set_facecolor("#1a0a14")
        ax.tick_params(colors="#c084fc", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#5a1a3a")
        ax.fill_between(xs, data, alpha=0.2, color=col)
        ax.plot(xs, data, color=col, lw=1.5)
        if len(data) >= 10:
            w  = min(20, len(data))
            ma = np.convolve(data, np.ones(w)/w, mode="valid")
            ax.plot(xs[w-1:], ma, color="#ffb3d9", lw=2.2, label=f"avg({w})")
            ax.legend(fontsize=8, facecolor="#2b1020", labelcolor="#ffe6f2")
        ax.set_title(title, color="#ffe6f2", fontsize=11, pad=6)
        ax.set_xlabel("Episode", color="#c084fc")
    fig.suptitle(f"✦  Tennis-v5  ·  CnnPolicy  ·  {name}  ·  Kariza",
                 color="#ff85c0", fontsize=12)
    fig.tight_layout(pad=1.5)
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Chart -> {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Build config from DEFAULTS + overrides
# ──────────────────────────────────────────────────────────────────────────────
def build_config(overrides: dict) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(overrides)
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Train one experiment
# ──────────────────────────────────────────────────────────────────────────────
def train_one(name: str, cfg: dict, out_dir: str) -> dict:
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  {name}  |  {ENV_ID}  |  {cfg['policy']}")
    print(sep)
    print(f"  lr={cfg['learning_rate']}  gamma={cfg['gamma']}  "
          f"batch={cfg['batch_size']}  n_envs={cfg['n_envs']}")
    print(f"  eps: {cfg['epsilon_start']} -> {cfg['epsilon_end']}  "
          f"decay={cfg['epsilon_decay']}")
    print(f"  timesteps={cfg['total_timesteps']:,}  "
          f"buffer={cfg['buffer_size']:,}\n")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    env   = make_env(cfg["n_envs"])
    model = DQN(
        cfg["policy"], env,
        learning_rate           = cfg["learning_rate"],
        gamma                   = cfg["gamma"],
        batch_size              = cfg["batch_size"],
        exploration_initial_eps = cfg["epsilon_start"],
        exploration_final_eps   = cfg["epsilon_end"],
        exploration_fraction    = cfg["epsilon_decay"],
        buffer_size             = cfg["buffer_size"],
        learning_starts         = cfg["learning_starts"],
        target_update_interval  = cfg["target_update"],
        train_freq              = cfg["train_freq"],
        optimize_memory_usage   = False,
        verbose                 = 0,
    )

    log_csv = os.path.join(out_dir, f"{name}_episodes.csv")
    cb = TrainCallback(cfg["total_timesteps"], log_csv,
                       save_freq=10_000, print_freq=5)

    t0 = time.time()
    model.learn(total_timesteps=cfg["total_timesteps"],
                callback=cb, log_interval=None)
    elapsed = time.time() - t0

    rewards = cb.ep_rewards
    n_ep     = len(rewards)
    mean_r   = round(float(np.mean(rewards)),       2) if rewards else 0.0
    max_r    = round(float(max(rewards)),            2) if rewards else 0.0
    last20   = round(float(np.mean(rewards[-20:])), 2) if len(rewards) >= 20 \
               else round(float(np.mean(rewards)),  2) if rewards else 0.0

    sps = cfg["total_timesteps"] / max(elapsed, 1)
    print(f"\n  Done {elapsed:.0f}s ({elapsed/60:.1f}m) | {sps:.0f} sps")
    print(f"  episodes={n_ep}  mean={mean_r}  max={max_r}  last20={last20}")

    # save model — naming matches assignment spec
    model_name = f"{name.lower()}_{cfg['policy'].lower()}"
    model_path = os.path.join(MODELS_DIR, f"{model_name}.zip")
    model.save(model_path)
    model.save(CHECKPOINT)
    print(f"  Model -> {model_path}")

    # save as best if beats previous
    prev_best = _get_best_score()
    if mean_r > prev_best:
        model.save(BEST_MODEL)
        _set_best_score(mean_r, name)
        print(f"  ★ NEW BEST → {BEST_MODEL}  "
              f"(mean={mean_r}  prev={prev_best})")

    # chart
    save_chart(rewards, cb.ep_lengths, out_dir, name)

    env.close()

    result = {
        "experiment":   name,
        "policy":       cfg["policy"],
        "mean_reward":  mean_r,
        "max_reward":   max_r,
        "last20_mean":  last20,
        "episodes":     n_ep,
        "time":         round(elapsed, 1),
        "hyperparams": {
            "learning_rate": cfg["learning_rate"],
            "gamma":         cfg["gamma"],
            "batch_size":    cfg["batch_size"],
            "epsilon_start": cfg["epsilon_start"],
            "epsilon_end":   cfg["epsilon_end"],
            "epsilon_decay": cfg["epsilon_decay"],
            "total_timesteps": cfg["total_timesteps"],
        },
        "model_path": model_path,
    }
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Save results JSON — matches assignment spec format
# ──────────────────────────────────────────────────────────────────────────────
def save_results_json(results: list, tag: str):
    path = f"results_{tag}.json"
    # format to match assignment spec
    output = []
    for r in results:
        output.append({
            "experiment":  r["experiment"],
            "policy":      r["policy"],
            "mean_reward": r["mean_reward"],
            "max_reward":  r["max_reward"],
            "episodes":    r["episodes"],
            "time":        r["time"],
        })
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results JSON -> {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Summary chart
# ──────────────────────────────────────────────────────────────────────────────
def save_summary_chart(results: list, out_dir: str):
    if len(results) < 2:
        return
    names   = [r["experiment"] for r in results]
    means   = [r["mean_reward"] for r in results]
    maxes   = [r["max_reward"]  for r in results]
    last20s = [r["last20_mean"] for r in results]

    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#2b1020")
    ax.set_facecolor("#1a0a14")
    ax.tick_params(colors="#7a7ab0", labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_edgecolor("#5a1a3a")

    x = np.arange(len(names))
    w = 0.28
    ax.bar(x-w, means,   width=w, color="#b57bee", alpha=0.9, label="Mean reward")
    ax.bar(x,   last20s, width=w, color="#ff85c0", alpha=0.9, label="Last-20 mean")
    ax.bar(x+w, maxes,   width=w, color="#c94b8a", alpha=0.9, label="Max reward")

    short = [n.replace("Exp","E").replace("_"," ")[:14] for n in names]
    ax.set_xticks(x)
    ax.set_xticklabels(short, color="#f9a8d4", fontsize=7.5, rotation=15)
    ax.set_ylabel("Reward", color="#c084fc")
    ax.set_title(
        f"✦  Kariza's Tennis Lab  ·  CnnPolicy  ·  All 10 Experiments",
        color="#ffe6f2", fontsize=12, pad=10)
    ax.legend(fontsize=9, facecolor="#2b1020", labelcolor="#ffe6f2")

    # mark best
    best_idx = int(np.argmax(means))
    ax.get_xticklabels()[best_idx].set_color("#ffb3d9")
    ax.get_xticklabels()[best_idx].set_fontweight("bold")

    fig.tight_layout(pad=1.5)
    path = os.path.join(out_dir, "summary_all_experiments.png")
    fig.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Summary chart -> {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Train CnnPolicy DQN on ALE/Tennis-v5  —  Kariza",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--exp",        default=None,
                   help="Run a specific experiment by name")
    p.add_argument("--all",        action="store_true",
                   help="Run all experiments in EXPERIMENTS list")
    p.add_argument("--policy",     default="CnnPolicy",
                   choices=["CnnPolicy"])
    p.add_argument("--timesteps",  type=int,   default=None,
                   help="Override total_timesteps for all runs")
    p.add_argument("--lr",         type=float, default=None, dest="learning_rate")
    p.add_argument("--gamma",      type=float, default=None)
    p.add_argument("--batch",      type=int,   default=None, dest="batch_size")
    p.add_argument("--eps-start",  type=float, default=None, dest="epsilon_start")
    p.add_argument("--eps-end",    type=float, default=None, dest="epsilon_end")
    p.add_argument("--eps-decay",  type=float, default=None, dest="epsilon_decay")
    p.add_argument("--n-envs",     type=int,   default=4,    dest="n_envs")
    p.add_argument("--save-freq",  type=int,   default=10_000, dest="save_freq")
    args = p.parse_args()

    tag     = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("logs", tag)
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 64)
    print("  TENNIS-V5  |  DQN  |  CnnPolicy  |  Kariza")
    print("=" * 64)
    print(f"  Environment : {ENV_ID}")
    print(f"  Policy      : {args.policy}")
    print(f"  Actions     : Discrete(18)  — {len(ACTION_NAMES)} actions")
    print(f"  n_envs      : {args.n_envs}")
    print("=" * 64)

    results = []

    if args.all:
        # run all experiments from EXPERIMENTS list
        for exp in EXPERIMENTS:
            cfg = build_config(exp["config_changes"])
            cfg["policy"]  = args.policy
            cfg["n_envs"]  = args.n_envs
            if args.timesteps:
                cfg["total_timesteps"] = args.timesteps
            result = train_one(exp["name"], cfg, out_dir)
            results.append(result)

        # save results JSON
        json_path = save_results_json(results, tag)

        # summary chart
        save_summary_chart(results, out_dir)

        # print table
        print("\n" + "=" * 64)
        print("  ALL EXPERIMENTS COMPLETE")
        print("=" * 64)
        print(f"\n  {'Experiment':<30} {'Mean':>7} {'Max':>7} {'Last20':>7} {'Eps':>5}")
        print(f"  {'─'*30} {'─'*7} {'─'*7} {'─'*7} {'─'*5}")
        for r in sorted(results, key=lambda x: x["mean_reward"], reverse=True):
            marker = " ★" if r == max(results, key=lambda x: x["mean_reward"]) else ""
            print(f"  {r['experiment']:<30} "
                  f"{r['mean_reward']:>7.2f} "
                  f"{r['max_reward']:>7.2f} "
                  f"{r['last20_mean']:>7.2f} "
                  f"{r['episodes']:>5}{marker}")

        if results:
            best = max(results, key=lambda x: x["mean_reward"])
            print(f"\n  ★ Best: {best['experiment']}  mean={best['mean_reward']}")
            print(f"  Model: {BEST_MODEL}")
        print(f"\n  Results JSON : {json_path}")
        print(f"  Models dir   : {MODELS_DIR}/")
        print("=" * 64 + "\n")

    elif args.exp:
        # run specific named experiment
        exp = next((e for e in EXPERIMENTS if e["name"] == args.exp), None)
        if exp is None:
            print(f"  Experiment '{args.exp}' not found.")
            print(f"  Available: {[e['name'] for e in EXPERIMENTS]}")
            return
        cfg = build_config(exp["config_changes"])
        cfg["policy"] = args.policy
        cfg["n_envs"] = args.n_envs
        if args.timesteps:
            cfg["total_timesteps"] = args.timesteps
        result = train_one(exp["name"], cfg, out_dir)
        save_results_json([result], tag)

    else:
        # single run with CLI flags
        overrides = {}
        for key in ["learning_rate","gamma","batch_size","epsilon_start",
                    "epsilon_end","epsilon_decay","n_envs"]:
            val = getattr(args, key, None)
            if val is not None:
                overrides[key] = val
        if args.timesteps:
            overrides["total_timesteps"] = args.timesteps
        overrides["policy"] = args.policy

        cfg    = build_config(overrides)
        result = train_one("CLI_Run", cfg, out_dir)
        save_results_json([result], tag)


if __name__ == "__main__":
    main()
