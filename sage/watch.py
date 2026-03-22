"""
watch.py — ALE/Tennis-v5  |  Student: Sage  |  Visual Game Player
──────────────────────────────────────────────────────────────────
Opens a Tkinter GUI window showing the trained DQN agent playing
Tennis in real time. Select any of the 10 trained models from the
dropdown and watch the agent play.

Usage:
    python3 watch.py                           # opens GUI with best model
    python3 watch.py --model models/exp7_latestart_cnnpolicy.zip
    python3 watch.py --episodes 10            # auto-stop after N episodes
"""

import argparse
import os
import threading
import time
import tkinter as tk
from tkinter import ttk, font as tkfont

import numpy as np

try:
    from PIL import Image, ImageTk
    PIL_OK = True
except ImportError:
    PIL_OK = False

import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

gym.register_envs(ale_py)

# ── Paths ─────────────────────────────────────────────────────────────────────
SAGE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SAGE_DIR, "models")
BEST_MODEL = os.path.join(SAGE_DIR, "dqn_model.zip")

ENV_ID = "ALE/Tennis-v5"

# ── Colour palette (blue/teal — Sage's theme) ─────────────────────────────────
BG       = "#0a1628"
BG2      = "#0f2040"
ACCENT   = "#00c9ff"
ACCENT2  = "#00b894"
FG       = "#e0f7fa"
FG_DIM   = "#607d8b"
RED      = "#ef5350"
GOLD     = "#ffd700"
PANEL_BG = "#071220"

# ── Model registry ────────────────────────────────────────────────────────────
EXPERIMENT_NOTES = {
    "exp1_cnn_baseline":          "Baseline  |  lr=2.5e-4  buf=50k  tgt=1000",
    "exp2_largebuffer":           "Large Buffer 100k  |  lr=2.5e-4  tgt=1000  ★ BEST",
    "exp3_smallbuffer":           "Small Buffer 10k  |  recency-biased",
    "exp4_frequenttargetupdate":  "Target update every 250 steps",
    "exp5_raretargetupdate":      "Target update every 5000 steps",
    "exp6_hightrainfreq":         "Train every step (train_freq=1)  |  lr=1e-4",
    "exp7_latestart":             "Late start (25k steps explore)  |  max +1.0 !!",
    "exp8_conservativeexplore":   "Conservative ε_end=0.15, decay=0.40",
    "exp9_fastconverge":          "Fast converge  |  lr=5e-4, batch=64, tgt=500",
    "exp10_bestcombo":            "Best-combo guess  |  lr=2e-4, buf=75k, tgt=1500",
}

RESULTS = {
    "exp1_cnn_baseline":          {"mean": -22.14, "max": -1.0,  "last20": -19.55},
    "exp2_largebuffer":           {"mean": -20.18, "max": -1.0,  "last20": -19.20},
    "exp3_smallbuffer":           {"mean": -22.96, "max": -8.0,  "last20": -20.35},
    "exp4_frequenttargetupdate":  {"mean": -22.46, "max": -1.0,  "last20": -18.60},
    "exp5_raretargetupdate":      {"mean": -20.94, "max": -1.0,  "last20": -17.80},
    "exp6_hightrainfreq":         {"mean": -22.98, "max": -3.0,  "last20": -21.60},
    "exp7_latestart":             {"mean": -20.50, "max":  1.0,  "last20": -17.40},
    "exp8_conservativeexplore":   {"mean": -23.80, "max": -19.0, "last20": -23.65},
    "exp9_fastconverge":          {"mean": -23.01, "max": -8.0,  "last20": -20.20},
    "exp10_bestcombo":            {"mean": -22.01, "max": -1.0,  "last20": -18.15},
}


def discover_models() -> list[tuple[str, str]]:
    """Returns list of (display_label, full_path) for all .zip models found."""
    entries = []
    if os.path.exists(BEST_MODEL):
        entries.append(("★ Best Model (dqn_model.zip)", BEST_MODEL))
    if os.path.isdir(MODELS_DIR):
        for fname in sorted(os.listdir(MODELS_DIR)):
            if fname.endswith(".zip"):
                key = fname.replace("_cnnpolicy.zip", "").replace("sage_", "")
                note = EXPERIMENT_NOTES.get(key, "")
                label = fname.replace("_cnnpolicy.zip", "").replace("_", " ").title()
                if note:
                    label = f"{label}  —  {note}"
                entries.append((label, os.path.join(MODELS_DIR, fname)))
    return entries


# ── Environment helpers ───────────────────────────────────────────────────────
def make_render_env() -> VecFrameStack:
    """Env with rgb_array rendering so we can grab frames."""
    env = make_atari_env(
        ENV_ID, n_envs=1, seed=0,
        vec_env_cls=DummyVecEnv,
        env_kwargs={"render_mode": "rgb_array"},
    )
    return VecFrameStack(env, n_stack=4)


def get_frame(env: VecFrameStack) -> np.ndarray | None:
    """Grab the current full-colour game frame (210×160×3)."""
    try:
        # SB3 DummyVecEnv.render() → list of rgb arrays when render_mode=rgb_array
        frames = env.render()
        if frames is None:
            return None
        if isinstance(frames, (list, tuple)):
            frames = frames[0]
        if isinstance(frames, np.ndarray):
            if frames.ndim == 4:          # (n_envs, H, W, C)
                frames = frames[0]
            return frames.astype(np.uint8)
    except Exception:
        pass
    # Fallback: try to reach the underlying ALE env
    try:
        inner = env.venv.envs[0]
        while hasattr(inner, "env"):
            inner = inner.env
        f = inner.render()
        if f is not None:
            return np.array(f, dtype=np.uint8)
    except Exception:
        pass
    return None


# ── Main GUI ──────────────────────────────────────────────────────────────────
class TennisWatcher:
    DISPLAY_W = 560
    DISPLAY_H = 420
    PANEL_W   = 320

    def __init__(self, root: tk.Tk, initial_model: str, max_episodes: int = 0):
        self.root         = root
        self.max_episodes = max_episodes

        self.env:   VecFrameStack | None = None
        self.model: DQN | None           = None
        self._model_path = ""

        self._running   = False
        self._paused    = False
        self._thread:   threading.Thread | None = None
        self._stop_evt  = threading.Event()

        self.episode     = 0
        self.ep_reward   = 0.0
        self.ep_step     = 0
        self.total_steps = 0
        self.all_rewards: list[float] = []
        self._speed_ms   = 30          # ms between frames

        self._build_ui()
        self._populate_model_list()

        # select initial model
        models = discover_models()
        initial_idx = 0
        for i, (_, path) in enumerate(models):
            if os.path.abspath(path) == os.path.abspath(initial_model):
                initial_idx = i
                break
        self.model_var.set(models[initial_idx][0] if models else "")
        self._on_model_change()

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        self.root.title("Sage · DQN Tennis Watcher · CnnPolicy")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        # ── Title bar ─────────────────────────────────────────────────────────
        title_frame = tk.Frame(self.root, bg=BG, pady=6)
        title_frame.pack(fill=tk.X)
        tk.Label(
            title_frame, text="SAGE  ·  DQN Tennis  ·  CnnPolicy  ·  ALE/Tennis-v5",
            bg=BG, fg=ACCENT, font=("Courier New", 13, "bold"),
        ).pack()

        # ── Main body ─────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # Left: game canvas
        left = tk.Frame(body, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH)

        self.canvas = tk.Canvas(
            left, width=self.DISPLAY_W, height=self.DISPLAY_H,
            bg="#000010", highlightthickness=2, highlightbackground=ACCENT,
        )
        self.canvas.pack()
        self._blank_canvas()

        # Right: control panel
        right = tk.Frame(body, bg=PANEL_BG, width=self.PANEL_W,
                         padx=12, pady=10)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right.pack_propagate(False)
        self._build_panel(right)

    def _blank_canvas(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(
            0, 0, self.DISPLAY_W, self.DISPLAY_H, fill="#000010", outline="")
        self.canvas.create_text(
            self.DISPLAY_W // 2, self.DISPLAY_H // 2,
            text="Loading model…", fill=FG_DIM,
            font=("Courier New", 14),
        )

    def _build_panel(self, parent):
        def section(label):
            tk.Label(parent, text=label, bg=PANEL_BG, fg=ACCENT,
                     font=("Courier New", 9, "bold")).pack(anchor=tk.W, pady=(10, 2))
            tk.Frame(parent, bg=ACCENT, height=1).pack(fill=tk.X)

        def stat_row(label, textvariable, color=FG):
            row = tk.Frame(parent, bg=PANEL_BG)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=f"{label:<14}", bg=PANEL_BG,
                     fg=FG_DIM, font=("Courier New", 9)).pack(side=tk.LEFT)
            tk.Label(row, textvariable=textvariable, bg=PANEL_BG,
                     fg=color, font=("Courier New", 10, "bold")).pack(side=tk.LEFT)

        # ── Model selector ────────────────────────────────────────────────────
        section("MODEL")
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            parent, textvariable=self.model_var, state="readonly",
            font=("Courier New", 8), width=36,
        )
        self.model_combo.pack(fill=tk.X, pady=(4, 0))
        self.model_combo.bind("<<ComboboxSelected>>", lambda _: self._on_model_change())

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=BG2, background=BG2,
                        foreground=FG, selectbackground=BG2,
                        selectforeground=ACCENT, arrowcolor=ACCENT)

        # ── Live stats ────────────────────────────────────────────────────────
        section("LIVE STATS")
        self.v_episode  = tk.StringVar(value="0")
        self.v_reward   = tk.StringVar(value="—")
        self.v_step     = tk.StringVar(value="0")
        self.v_best_ep  = tk.StringVar(value="—")
        self.v_mean_all = tk.StringVar(value="—")
        self.v_status   = tk.StringVar(value="Loading…")

        stat_row("Episode",   self.v_episode,  ACCENT)
        stat_row("Ep Reward", self.v_reward,   GOLD)
        stat_row("Ep Steps",  self.v_step,     FG)
        stat_row("Best Ep",   self.v_best_ep,  ACCENT2)
        stat_row("Mean All",  self.v_mean_all, FG)
        stat_row("Status",    self.v_status,   ACCENT2)

        # ── Training results ──────────────────────────────────────────────────
        section("TRAINING RESULTS (200k steps)")
        self.v_train_mean  = tk.StringVar(value="—")
        self.v_train_max   = tk.StringVar(value="—")
        self.v_train_last20 = tk.StringVar(value="—")
        stat_row("Train mean",  self.v_train_mean,   FG)
        stat_row("Train max",   self.v_train_max,    ACCENT2)
        stat_row("Last-20 avg", self.v_train_last20, FG)

        # ── Speed control ─────────────────────────────────────────────────────
        section("PLAYBACK SPEED")
        speed_row = tk.Frame(parent, bg=PANEL_BG)
        speed_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(speed_row, text="Slow", bg=PANEL_BG,
                 fg=FG_DIM, font=("Courier New", 8)).pack(side=tk.LEFT)
        self.speed_slider = tk.Scale(
            speed_row, from_=120, to=5, orient=tk.HORIZONTAL,
            bg=PANEL_BG, fg=FG, troughcolor=BG2,
            highlightthickness=0, showvalue=False,
            command=self._on_speed_change,
        )
        self.speed_slider.set(30)
        self.speed_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(speed_row, text="Fast", bg=PANEL_BG,
                 fg=FG_DIM, font=("Courier New", 8)).pack(side=tk.LEFT)

        # ── Buttons ───────────────────────────────────────────────────────────
        section("CONTROLS")
        btn_cfg = dict(bg=BG2, fg=FG, font=("Courier New", 10, "bold"),
                       relief=tk.FLAT, cursor="hand2", pady=5, padx=8,
                       activebackground=ACCENT, activeforeground=BG)

        btn_row1 = tk.Frame(parent, bg=PANEL_BG)
        btn_row1.pack(fill=tk.X, pady=(6, 2))

        self.btn_play = tk.Button(
            btn_row1, text="▶  PLAY", **btn_cfg,
            command=self._toggle_pause,
        )
        self.btn_play.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        tk.Button(
            btn_row1, text="↺  NEW EP", **btn_cfg,
            command=self._new_episode,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Button(
            parent, text="■  STOP", command=self._stop,
            **{**btn_cfg, "fg": RED},
        ).pack(fill=tk.X, pady=(2, 0))

        # ── Footer info ───────────────────────────────────────────────────────
        tk.Frame(parent, bg=PANEL_BG).pack(fill=tk.Y, expand=True)
        tk.Label(
            parent,
            text="n_envs=1  ·  200k steps\nCnnPolicy  ·  ALE/Tennis-v5",
            bg=PANEL_BG, fg=FG_DIM, font=("Courier New", 7),
            justify=tk.CENTER,
        ).pack(pady=(4, 0))

    # ── Model management ──────────────────────────────────────────────────────
    def _populate_model_list(self):
        self._models = discover_models()
        labels = [label for label, _ in self._models]
        self.model_combo["values"] = labels

    def _on_model_change(self, *_):
        label = self.model_var.get()
        path  = next((p for l, p in self._models if l == label), None)
        if path and path != self._model_path:
            self._stop()
            self.v_status.set("Loading…")
            self._blank_canvas()
            threading.Thread(target=self._load_and_start,
                             args=(path,), daemon=True).start()

    def _load_and_start(self, path: str):
        try:
            if self.env is not None:
                try:
                    self.env.close()
                except Exception:
                    pass
            self.env = make_render_env()
            self.model = DQN.load(path, env=self.env)
            self._model_path = path

            # update training stats in UI
            key = os.path.basename(path).replace("_cnnpolicy.zip", "").replace("sage_", "")
            res = RESULTS.get(key, {})
            self.root.after(0, lambda: self.v_train_mean.set(
                f"{res.get('mean', '—'):.2f}" if res else "—"))
            self.root.after(0, lambda: self.v_train_max.set(
                f"{res.get('max', '—'):.2f}" if res else "—"))
            self.root.after(0, lambda: self.v_train_last20.set(
                f"{res.get('last20', '—'):.2f}" if res else "—"))

            self.root.after(0, lambda: self.v_status.set("Ready"))
            self._start_episode()
        except Exception as e:
            self.root.after(0, lambda: self.v_status.set(f"Error: {e}"))

    # ── Game loop (background thread) ─────────────────────────────────────────
    def _start_episode(self):
        self._stop_evt.clear()
        self._paused  = False
        self._running = True
        self.root.after(0, lambda: self.btn_play.config(text="⏸  PAUSE"))
        self._thread = threading.Thread(target=self._game_loop, daemon=True)
        self._thread.start()

    def _game_loop(self):
        if self.env is None or self.model is None:
            return

        obs = self.env.reset()
        self.episode   += 1
        self.ep_reward  = 0.0
        self.ep_step    = 0
        self.root.after(0, self._update_stats)

        while not self._stop_evt.is_set():
            while self._paused and not self._stop_evt.is_set():
                time.sleep(0.05)
            if self._stop_evt.is_set():
                break

            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)

            self.ep_reward  += float(reward[0])
            self.ep_step    += 1
            self.total_steps += 1

            # grab and display frame
            frame = get_frame(self.env)
            if frame is not None:
                self.root.after(0, lambda f=frame: self._display_frame(f))

            self.root.after(0, self._update_stats)

            time.sleep(self._speed_ms / 1000.0)

            if done[0]:
                self.all_rewards.append(self.ep_reward)
                if self.max_episodes and self.episode >= self.max_episodes:
                    self.root.after(0, lambda: self.v_status.set("Done"))
                    return
                # small pause between episodes
                time.sleep(0.6)
                obs = self.env.reset()
                self.episode  += 1
                self.ep_reward = 0.0
                self.ep_step   = 0
                self.root.after(0, self._update_stats)

        self._running = False
        self.root.after(0, lambda: self.btn_play.config(text="▶  PLAY"))

    def _display_frame(self, frame: np.ndarray):
        if not PIL_OK:
            return
        try:
            img  = Image.fromarray(frame)
            img  = img.resize((self.DISPLAY_W, self.DISPLAY_H), Image.NEAREST)
            photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas._photo_ref = photo       # keep ref — prevents GC
        except Exception:
            pass

    def _update_stats(self):
        self.v_episode.set(str(self.episode))
        self.v_reward.set(f"{self.ep_reward:.1f}")
        self.v_step.set(f"{self.ep_step:,}")
        if self.all_rewards:
            self.v_best_ep.set(f"{max(self.all_rewards):.1f}")
            self.v_mean_all.set(f"{np.mean(self.all_rewards):.2f}")
        status = "Paused" if self._paused else ("Playing" if self._running else "Stopped")
        self.v_status.set(status)

    # ── Controls ──────────────────────────────────────────────────────────────
    def _toggle_pause(self):
        if not self._running:
            self._start_episode()
            return
        self._paused = not self._paused
        self.btn_play.config(text="▶  PLAY" if self._paused else "⏸  PAUSE")
        self.v_status.set("Paused" if self._paused else "Playing")

    def _new_episode(self):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=1.5)
        self._running = False
        self._start_episode()

    def _stop(self):
        self._stop_evt.set()
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.5)
        self.btn_play.config(text="▶  PLAY")
        self.v_status.set("Stopped")

    def _on_speed_change(self, val):
        self._speed_ms = int(val)

    def on_close(self):
        self._stop()
        if self.env:
            try:
                self.env.close()
            except Exception:
                pass
        self.root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    if not PIL_OK:
        print("\n  Missing Pillow — install it first:")
        print("  pip install Pillow\n")
        return

    p = argparse.ArgumentParser(
        description="Visual DQN Tennis player — Sage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",    default=BEST_MODEL,
                   help="Path to .zip model to load on startup")
    p.add_argument("--episodes", type=int, default=0,
                   help="Stop after N episodes (0 = run forever)")
    args = p.parse_args()

    if not os.path.exists(args.model):
        # fall back to first model found
        found = discover_models()
        if not found:
            print("  No trained models found — run train.py --all first.")
            return
        args.model = found[0][1]
        print(f"  dqn_model.zip not found — loading: {args.model}")

    root = tk.Tk()
    app  = TennisWatcher(root, initial_model=args.model,
                         max_episodes=args.episodes)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
