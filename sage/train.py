"""
train.py — ALE/Tennis-v5  |  Student: Sage  |  Policy: CnnPolicy
──────────────────────────────────────────────────────────────────
Train DQN agent on Atari Tennis using CnnPolicy.

KEY DIFFERENCES FROM KARIZA'S APPROACH:
  • Single environment (n_envs=1) — sequential, focused experience collection
    vs Kariza's 4 parallel envs (VecFrameStack / make_atari_env)
  • 200,000 timesteps per experiment — 2× Kariza's 100k
  • Hyperparameter axes explored: buffer_size, target_update_interval,
    train_freq, learning_starts — dimensions Kariza did NOT vary
  • Best model saved as dqn_model.zip (assignment spec)
  • Outputs a full Markdown + CSV hyperparameter table after all runs

HOW TO USE:
    python3 train.py                          # runs Exp1_CNN_Baseline only
    python3 train.py --exp Exp3_FreqUpdate    # run specific experiment by name
    python3 train.py --all                    # run all 10 experiments in sequence
    python3 train.py --list                   # print all experiment names
    python3 train.py --lr 2e-4 --timesteps 150000   # one-off CLI run
"""

import argparse
import csv
import json
import os
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
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

gym.register_envs(ale_py)

# ── Paths & constants ─────────────────────────────────────────────────────────
ENV_ID        = "ALE/Tennis-v5"
SAGE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR    = os.path.join(SAGE_DIR, "models")
LOGS_BASE_DIR = os.path.join(SAGE_DIR, "logs")
BEST_MODEL    = os.path.join(SAGE_DIR, "dqn_model.zip")       # assignment spec name
CHECKPOINT    = os.path.join(SAGE_DIR, "dqn_latest.zip")
BEST_SCORE_F  = os.path.join(SAGE_DIR, "best_score.json")
HYPER_CSV     = os.path.join(SAGE_DIR, "hyperparameter_experiments.csv")

# ── Default hyperparameters ───────────────────────────────────────────────────
# NOTE: Sage uses n_envs=1 (single env) — Kariza used n_envs=4
DEFAULTS = {
    "policy":           "CnnPolicy",
    "learning_rate":    2.5e-4,
    "gamma":            0.99,
    "batch_size":       32,
    "buffer_size":      50_000,
    "epsilon_start":    1.0,
    "epsilon_end":      0.05,
    "epsilon_decay":    0.20,       # fraction of total_timesteps for ε decay
    "total_timesteps":  200_000,
    "n_envs":           1,          # ← single env (Kariza used 4)
    "learning_starts":  10_000,
    "target_update":    1_000,
    "train_freq":       4,
}

# ──────────────────────────────────────────────────────────────────────────────
# 10 EXPERIMENTS — focus on buffer, target_update, train_freq, learning_starts
# (axes Kariza did NOT vary; only Exp10 overlaps slightly on lr/batch)
# ──────────────────────────────────────────────────────────────────────────────
EXPERIMENTS = [
    # 1 — Standard CNN baseline for Sage (different defaults from Kariza's)
    {
        "name": "Exp1_CNN_Baseline",
        "note": "Standard baseline — lr=2.5e-4, γ=0.99, batch=32, buf=50k",
        "config_changes": {},
    },
    # 2 — Double the replay buffer for more diverse experience sampling
    {
        "name": "Exp2_LargeBuffer",
        "note": "Large replay buffer (100k) — more diverse past experiences",
        "config_changes": {
            "buffer_size": 100_000,
        },
    },
    # 3 — Small buffer: agent learns mostly from recent experience
    {
        "name": "Exp3_SmallBuffer",
        "note": "Small replay buffer (10k) — recency-biased learning",
        "config_changes": {
            "buffer_size": 10_000,
            "learning_starts": 2_000,
        },
    },
    # 4 — Frequent target network updates → faster adaptation, more instability
    {
        "name": "Exp4_FrequentTargetUpdate",
        "note": "Target net updated every 250 steps (Kariza used 1000)",
        "config_changes": {
            "target_update": 250,
        },
    },
    # 5 — Rare target updates → more stable Q-targets, slower adaptation
    {
        "name": "Exp5_RareTargetUpdate",
        "note": "Target net updated every 5000 steps — very stable targets",
        "config_changes": {
            "target_update": 5_000,
        },
    },
    # 6 — Train after every step (train_freq=1) — maximum gradient updates
    {
        "name": "Exp6_HighTrainFreq",
        "note": "Train every env step (train_freq=1) — most updates per timestep",
        "config_changes": {
            "train_freq":    1,
            "learning_rate": 1e-4,   # lower LR to compensate for more updates
        },
    },
    # 7 — Extended warm-up: explore 25k steps before any learning begins
    {
        "name": "Exp7_LateStart",
        "note": "learning_starts=25k — fill buffer with diverse plays first",
        "config_changes": {
            "learning_starts": 25_000,
            "buffer_size":     75_000,
        },
    },
    # 8 — Conservative exploration: higher ε_end + very slow ε decay
    {
        "name": "Exp8_ConservativeExplore",
        "note": "ε_end=0.15, decay over 40% of steps — sustained exploration",
        "config_changes": {
            "epsilon_end":   0.15,
            "epsilon_decay": 0.40,
            "buffer_size":   75_000,
        },
    },
    # 9 — Faster convergence attempt: higher LR, larger batch, quicker target sync
    {
        "name": "Exp9_FastConverge",
        "note": "lr=5e-4, batch=64, target_update=500, train_freq=2",
        "config_changes": {
            "learning_rate": 5e-4,
            "batch_size":    64,
            "target_update": 500,
            "train_freq":    2,
        },
    },
    # 10 — Best-guess combo based on observations from Exp1–9
    {
        "name": "Exp10_BestCombo",
        "note": "lr=2e-4, batch=64, buf=75k, target=1500, eps_decay=0.25, start=15k",
        "config_changes": {
            "learning_rate":   2e-4,
            "gamma":           0.99,
            "batch_size":      64,
            "buffer_size":     75_000,
            "target_update":   1_500,
            "epsilon_decay":   0.25,
            "learning_starts": 15_000,
        },
    },
]


# ── Best score tracker ────────────────────────────────────────────────────────
def _get_best_score() -> float:
    if os.path.exists(BEST_SCORE_F):
        with open(BEST_SCORE_F) as f:
            return json.load(f).get("mean_reward", -999.0)
    return -999.0


def _set_best_score(mean_reward: float, exp_name: str):
    with open(BEST_SCORE_F, "w") as f:
        json.dump({
            "mean_reward": mean_reward,
            "experiment":  exp_name,
            "saved_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)


# ── Environment builder ───────────────────────────────────────────────────────
def make_env(n_envs: int = 1, seed: int = 0) -> VecFrameStack:
    """
    Sage's env: make_atari_env with n_envs=1 (single env, DummyVecEnv).
    Applies standard Atari preprocessing: NoopReset, MaxAndSkip,
    EpisodicLife, FireReset, ClipReward, WarpFrame(84x84), VecFrameStack(4).
    Obs shape: (1, 4, 84, 84) — suitable for CnnPolicy.

    Key difference from Kariza: n_envs=1 here, Kariza used n_envs=4.
    Single env means strictly sequential experience; slower wall-clock speed
    but cleaner episode boundaries and lower memory footprint.
    """
    env = make_atari_env(
        ENV_ID,
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv,
        seed=seed,
    )
    return VecFrameStack(env, n_stack=4)


# ── Callback ──────────────────────────────────────────────────────────────────
class SageCallback(BaseCallback):
    """Tracks per-episode rewards/lengths, logs to CSV, prints progress."""

    def __init__(self, total_steps: int, log_csv: str,
                 save_freq: int = 10_000, print_freq: int = 5):
        super().__init__()
        self.total_steps = total_steps
        self.log_csv     = log_csv
        self.save_freq   = save_freq
        self.print_freq  = print_freq
        self.ep_rewards: list[float] = []
        self.ep_lengths: list[int]   = []
        self._ep_count  = 0
        self._last_save = 0
        self._start     = time.time()
        os.makedirs(os.path.dirname(log_csv), exist_ok=True)
        with open(log_csv, "w", newline="") as fh:
            csv.writer(fh).writerow(
                ["episode", "reward", "length", "elapsed_s", "steps"])

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
                with open(self.log_csv, "a", newline="") as fh:
                    csv.writer(fh).writerow([
                        self._ep_count, r, l,
                        round(time.time() - self._start, 1),
                        self.num_timesteps,
                    ])
                if self._ep_count % self.print_freq == 0:
                    recent = self.ep_rewards[-self.print_freq:]
                    sps    = self.num_timesteps / max(time.time() - self._start, 1)
                    pct    = self.num_timesteps / self.total_steps * 100
                    print(f"  Ep {self._ep_count:>4} | "
                          f"Steps {self.num_timesteps:>9,} ({pct:.0f}%) | "
                          f"SPS {sps:>5.0f} | "
                          f"Recent mean={np.mean(recent):>7.2f}  "
                          f"Best={max(self.ep_rewards):>6.2f}")
        return True


# ── Per-experiment chart ──────────────────────────────────────────────────────
def save_chart(rewards: list, lengths: list, out_dir: str, name: str):
    """Blue/teal theme — clearly distinct from Kariza's pink/purple."""
    if not rewards:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), facecolor="#0d1f2d")
    xs = list(range(1, len(rewards) + 1))
    palette = [("#00c9ff", "#00e5ff"), ("#00b894", "#55efc4")]
    for ax, data, (col, ma_col), title in [
        (ax1, rewards, palette[0], "Episode Reward"),
        (ax2, lengths, palette[1], "Episode Length"),
    ]:
        ax.set_facecolor("#091520")
        ax.tick_params(colors="#7ec8e3", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1a4a6b")
        ax.fill_between(xs, data, alpha=0.18, color=col)
        ax.plot(xs, data, color=col, lw=1.4, alpha=0.85)
        if len(data) >= 10:
            w  = min(20, len(data))
            ma = np.convolve(data, np.ones(w) / w, mode="valid")
            ax.plot(xs[w - 1:], ma, color=ma_col, lw=2.2, label=f"avg({w})")
            ax.legend(fontsize=8, facecolor="#0d1f2d", labelcolor="#b2ebf2")
        ax.set_title(title, color="#b2ebf2", fontsize=11, pad=6)
        ax.set_xlabel("Episode", color="#7ec8e3")
    fig.suptitle(
        f"Tennis-v5  ·  CnnPolicy  ·  {name}  ·  Sage",
        color="#00e5ff", fontsize=12,
    )
    fig.tight_layout(pad=1.5)
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Chart  → {path}")


# ── Summary bar chart across all experiments ──────────────────────────────────
def save_summary_chart(results: list, out_dir: str):
    if len(results) < 2:
        return
    names   = [r["experiment"] for r in results]
    means   = [r["mean_reward"] for r in results]
    maxes   = [r["max_reward"]  for r in results]
    last20s = [r["last20_mean"] for r in results]

    fig, ax = plt.subplots(figsize=(15, 5), facecolor="#0d1f2d")
    ax.set_facecolor("#091520")
    ax.tick_params(colors="#7ec8e3", labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a4a6b")

    x, w = np.arange(len(names)), 0.28
    ax.bar(x - w, means,   width=w, color="#00c9ff", alpha=0.88, label="Mean reward")
    ax.bar(x,     last20s, width=w, color="#00b894", alpha=0.88, label="Last-20 mean")
    ax.bar(x + w, maxes,   width=w, color="#6c5ce7", alpha=0.88, label="Max reward")

    short = [n.replace("Exp", "E").replace("_", " ")[:16] for n in names]
    ax.set_xticks(x)
    ax.set_xticklabels(short, color="#b2ebf2", fontsize=7, rotation=18)
    ax.set_ylabel("Reward", color="#7ec8e3")
    ax.set_title(
        "Sage's Tennis Lab  ·  CnnPolicy  ·  n_envs=1  ·  All 10 Experiments",
        color="#b2ebf2", fontsize=12, pad=10,
    )
    ax.legend(fontsize=9, facecolor="#0d1f2d", labelcolor="#b2ebf2")

    best_idx = int(np.argmax(means))
    ax.get_xticklabels()[best_idx].set_color("#00e5ff")
    ax.get_xticklabels()[best_idx].set_fontweight("bold")

    fig.tight_layout(pad=1.5)
    path = os.path.join(out_dir, "summary_all_experiments.png")
    fig.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Summary chart → {path}")


# ── Hyperparameter table (CSV + Markdown) ─────────────────────────────────────
HYPER_CSV_HEADER = [
    "exp_num", "experiment", "lr", "gamma", "batch_size",
    "epsilon_start", "epsilon_end", "epsilon_decay",
    "buffer_size", "target_update", "train_freq", "learning_starts",
    "timesteps", "mean_reward", "max_reward", "last20_mean",
    "episodes", "time_s", "notes",
]

def _init_hyper_csv():
    os.makedirs(SAGE_DIR, exist_ok=True)
    with open(HYPER_CSV, "w", newline="") as fh:
        csv.writer(fh).writerow(HYPER_CSV_HEADER)


def _append_hyper_csv(exp_num: int, result: dict, cfg: dict, note: str):
    h = cfg
    with open(HYPER_CSV, "a", newline="") as fh:
        csv.writer(fh).writerow([
            exp_num,
            result["experiment"],
            h["learning_rate"],
            h["gamma"],
            h["batch_size"],
            h["epsilon_start"],
            h["epsilon_end"],
            h["epsilon_decay"],
            h["buffer_size"],
            h["target_update"],
            h["train_freq"],
            h["learning_starts"],
            h["total_timesteps"],
            result["mean_reward"],
            result["max_reward"],
            result["last20_mean"],
            result["episodes"],
            result["time"],
            note,
        ])


def print_hyper_table(results: list, configs: list, notes: list):
    """Print a formatted hyperparameter + results table to the console."""
    sep = "─" * 130
    print(f"\n{sep}")
    print("  SAGE — HYPERPARAMETER EXPERIMENT TABLE  |  CnnPolicy  |  ALE/Tennis-v5")
    print(sep)
    header = (
        f"  {'#':>2}  {'Experiment':<28}  {'LR':>8}  {'γ':>6}  "
        f"{'Bat':>4}  {'ε_end':>5}  {'ε_dec':>5}  "
        f"{'Buf':>6}  {'TgtUpd':>6}  {'TrFq':>4}  {'Strt':>5}  "
        f"{'Steps':>7}  {'Mean':>7}  {'Max':>7}  {'Last20':>7}  {'Eps':>4}  {'Time':>6}"
    )
    print(header)
    print(f"  {'─'*2}  {'─'*28}  {'─'*8}  {'─'*6}  "
          f"{'─'*4}  {'─'*5}  {'─'*5}  "
          f"{'─'*6}  {'─'*6}  {'─'*4}  {'─'*5}  "
          f"{'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*4}  {'─'*6}")
    best_mean = max(r["mean_reward"] for r in results)
    for i, (r, cfg, note) in enumerate(zip(results, configs, notes), 1):
        marker = " ★" if r["mean_reward"] == best_mean else "  "
        print(
            f"  {i:>2}  {r['experiment']:<28}  "
            f"{cfg['learning_rate']:>8.2e}  {cfg['gamma']:>6.3f}  "
            f"{cfg['batch_size']:>4}  {cfg['epsilon_end']:>5.2f}  "
            f"{cfg['epsilon_decay']:>5.2f}  "
            f"{cfg['buffer_size']:>6,}  {cfg['target_update']:>6}  "
            f"{cfg['train_freq']:>4}  {cfg['learning_starts']:>5,}  "
            f"{cfg['total_timesteps']:>7,}  "
            f"{r['mean_reward']:>7.2f}  {r['max_reward']:>7.2f}  "
            f"{r['last20_mean']:>7.2f}  {r['episodes']:>4}  "
            f"{r['time']:>5.0f}s{marker}"
        )
    print(sep)
    print(f"  ★ Best mean reward: {best_mean}  |  CSV → {HYPER_CSV}")
    print(sep + "\n")


# ── Build config from DEFAULTS + overrides ────────────────────────────────────
def build_config(overrides: dict) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(overrides)
    return cfg


# ── Train one experiment ──────────────────────────────────────────────────────
def train_one(name: str, cfg: dict, out_dir: str) -> dict:
    sep = "─" * 68
    print(f"\n{sep}")
    print(f"  {name}  |  {ENV_ID}  |  {cfg['policy']}")
    print(sep)
    print(f"  lr={cfg['learning_rate']}  γ={cfg['gamma']}  "
          f"batch={cfg['batch_size']}  n_envs={cfg['n_envs']}")
    print(f"  ε: {cfg['epsilon_start']} → {cfg['epsilon_end']}  "
          f"decay={cfg['epsilon_decay']}")
    print(f"  buffer={cfg['buffer_size']:,}  "
          f"target_update={cfg['target_update']}  "
          f"train_freq={cfg['train_freq']}  "
          f"learning_starts={cfg['learning_starts']:,}")
    print(f"  timesteps={cfg['total_timesteps']:,}\n")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    env = make_env(n_envs=cfg["n_envs"], seed=42)

    model = DQN(
        cfg["policy"], env,
        learning_rate           = cfg["learning_rate"],
        gamma                   = cfg["gamma"],
        batch_size              = cfg["batch_size"],
        buffer_size             = cfg["buffer_size"],
        exploration_initial_eps = cfg["epsilon_start"],
        exploration_final_eps   = cfg["epsilon_end"],
        exploration_fraction    = cfg["epsilon_decay"],
        learning_starts         = cfg["learning_starts"],
        target_update_interval  = cfg["target_update"],
        train_freq              = cfg["train_freq"],
        optimize_memory_usage   = False,
        verbose                 = 0,
    )

    log_csv = os.path.join(out_dir, f"{name}_episodes.csv")
    cb = SageCallback(cfg["total_timesteps"], log_csv,
                      save_freq=10_000, print_freq=5)

    t0 = time.time()
    model.learn(total_timesteps=cfg["total_timesteps"],
                callback=cb, log_interval=None)
    elapsed = time.time() - t0

    rewards  = cb.ep_rewards
    n_ep     = len(rewards)
    mean_r   = round(float(np.mean(rewards)),        2) if rewards else 0.0
    max_r    = round(float(max(rewards)),             2) if rewards else 0.0
    last20   = round(float(np.mean(rewards[-20:])),  2) if len(rewards) >= 20 \
               else round(float(np.mean(rewards)),   2) if rewards else 0.0

    sps = cfg["total_timesteps"] / max(elapsed, 1)
    print(f"\n  Done {elapsed:.0f}s ({elapsed/60:.1f}m) | {sps:.0f} sps")
    print(f"  episodes={n_ep}  mean={mean_r}  max={max_r}  last20={last20}")

    # save named model
    slug       = name.lower().replace(" ", "_")
    model_path = os.path.join(MODELS_DIR, f"{slug}_cnnpolicy.zip")
    model.save(model_path)
    model.save(CHECKPOINT)
    print(f"  Model  → {model_path}")

    # track best → dqn_model.zip (assignment spec)
    prev_best = _get_best_score()
    if mean_r > prev_best:
        model.save(BEST_MODEL)
        _set_best_score(mean_r, name)
        print(f"  ★ NEW BEST → dqn_model.zip  "
              f"(mean={mean_r}  prev={prev_best})")

    save_chart(rewards, cb.ep_lengths, out_dir, name)

    env.close()

    return {
        "experiment":  name,
        "policy":      cfg["policy"],
        "mean_reward": mean_r,
        "max_reward":  max_r,
        "last20_mean": last20,
        "episodes":    n_ep,
        "time":        round(elapsed, 1),
        "model_path":  model_path,
        "hyperparams": {
            "learning_rate":   cfg["learning_rate"],
            "gamma":           cfg["gamma"],
            "batch_size":      cfg["batch_size"],
            "epsilon_start":   cfg["epsilon_start"],
            "epsilon_end":     cfg["epsilon_end"],
            "epsilon_decay":   cfg["epsilon_decay"],
            "buffer_size":     cfg["buffer_size"],
            "target_update":   cfg["target_update"],
            "train_freq":      cfg["train_freq"],
            "learning_starts": cfg["learning_starts"],
            "total_timesteps": cfg["total_timesteps"],
            "n_envs":          cfg["n_envs"],
        },
    }


# ── Save results JSON ─────────────────────────────────────────────────────────
def save_results_json(results: list, tag: str) -> str:
    path = os.path.join(SAGE_DIR, f"results_sage_{tag}.json")
    output = []
    for r in results:
        output.append({
            "experiment":  r["experiment"],
            "policy":      r["policy"],
            "mean_reward": r["mean_reward"],
            "max_reward":  r["max_reward"],
            "last20_mean": r["last20_mean"],
            "episodes":    r["episodes"],
            "time":        r["time"],
            "hyperparams": r["hyperparams"],
        })
    with open(path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\n  Results JSON → {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Train CnnPolicy DQN on ALE/Tennis-v5  —  Sage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--exp",       default=None,
                   help="Run a specific experiment by name")
    p.add_argument("--all",       action="store_true",
                   help="Run all 10 experiments in sequence")
    p.add_argument("--list",      action="store_true",
                   help="Print all experiment names and exit")
    p.add_argument("--policy",    default="CnnPolicy",
                   choices=["CnnPolicy"])
    p.add_argument("--timesteps", type=int,   default=None,
                   help="Override total_timesteps")
    p.add_argument("--lr",        type=float, default=None, dest="learning_rate")
    p.add_argument("--gamma",     type=float, default=None)
    p.add_argument("--batch",     type=int,   default=None, dest="batch_size")
    p.add_argument("--buffer",    type=int,   default=None, dest="buffer_size")
    p.add_argument("--eps-end",   type=float, default=None, dest="epsilon_end")
    p.add_argument("--eps-decay", type=float, default=None, dest="epsilon_decay")
    p.add_argument("--tgt-upd",   type=int,   default=None, dest="target_update")
    p.add_argument("--train-freq",type=int,   default=None, dest="train_freq")
    p.add_argument("--starts",    type=int,   default=None, dest="learning_starts")
    p.add_argument("--n-envs",    type=int,   default=1,    dest="n_envs",
                   help="Number of parallel envs (default 1 — Sage's approach)")
    p.add_argument("--save-freq", type=int,   default=10_000)
    args = p.parse_args()

    if args.list:
        print("\n  Available experiments:")
        for i, e in enumerate(EXPERIMENTS, 1):
            print(f"  {i:>2}. {e['name']:<30}  —  {e['note']}")
        print()
        return

    tag     = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(LOGS_BASE_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 68)
    print("  TENNIS-V5  |  DQN  |  CnnPolicy  |  Sage")
    print("=" * 68)
    print(f"  Environment  : {ENV_ID}")
    print(f"  Policy       : CnnPolicy")
    print(f"  n_envs       : {args.n_envs}  (Kariza used 4; Sage uses single env)")
    print(f"  Timesteps    : {args.timesteps or DEFAULTS['total_timesteps']:,}  "
          f"(Kariza used 100k; Sage uses 200k)")
    print(f"  Focus axes   : buffer_size, target_update, train_freq, learning_starts")
    print("=" * 68)

    results, configs, notes = [], [], []

    def _run_exp(exp_dict):
        cfg = build_config(exp_dict["config_changes"])
        cfg["policy"] = args.policy
        cfg["n_envs"] = args.n_envs
        if args.timesteps:
            cfg["total_timesteps"] = args.timesteps
        result = train_one(exp_dict["name"], cfg, out_dir)
        results.append(result)
        configs.append(cfg)
        notes.append(exp_dict.get("note", ""))
        return result

    if args.all:
        _init_hyper_csv()
        for i, exp in enumerate(EXPERIMENTS, 1):
            r   = _run_exp(exp)
            cfg = configs[-1]
            _append_hyper_csv(i, r, cfg, exp.get("note", ""))

        json_path = save_results_json(results, tag)
        save_summary_chart(results, out_dir)
        print_hyper_table(results, configs, notes)

        if results:
            best = max(results, key=lambda x: x["mean_reward"])
            print(f"  ★ Best overall: {best['experiment']}  "
                  f"mean={best['mean_reward']}  max={best['max_reward']}")
            print(f"  Saved as: {BEST_MODEL}")
        print(f"\n  Results JSON       : {json_path}")
        print(f"  Hyperparameter CSV : {HYPER_CSV}")
        print(f"  Models dir         : {MODELS_DIR}/")
        print(f"  Charts dir         : {out_dir}/")
        print("=" * 68 + "\n")

    elif args.exp:
        exp = next((e for e in EXPERIMENTS if e["name"] == args.exp), None)
        if exp is None:
            print(f"\n  Experiment '{args.exp}' not found.")
            print(f"  Run with --list to see all options.")
            return
        _run_exp(exp)
        save_results_json(results, tag)
        print_hyper_table(results, configs, notes)

    else:
        # CLI one-off run
        overrides: dict = {}
        for key in ["learning_rate", "gamma", "batch_size", "buffer_size",
                    "epsilon_end", "epsilon_decay", "target_update",
                    "train_freq", "learning_starts", "n_envs"]:
            val = getattr(args, key, None)
            if val is not None:
                overrides[key] = val
        if args.timesteps:
            overrides["total_timesteps"] = args.timesteps
        overrides["policy"] = args.policy
        cfg    = build_config(overrides)
        result = train_one("CLI_Run", cfg, out_dir)
        save_results_json([result], tag)
        print_hyper_table([result], [cfg], ["CLI one-off run"])


if __name__ == "__main__":
    main()
