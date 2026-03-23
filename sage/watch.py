"""
watch.py — ALE/Tennis-v5  |  Student: Sage  |  Visual Game Player
──────────────────────────────────────────────────────────────────
Three play modes in one window:

  WATCH AI   — DQN agent plays, you observe
  PLAY HUMAN — YOU play with keyboard vs the ALE CPU
  AI HINT    — You play, but DQN's suggested action shown live

Keyboard controls (Human / Hint mode):
  Arrow keys / WASD  →  move player
  Space              →  SWING (hit the ball)
  Combine both       →  move + swing at the same time

Usage:
    python3 watch.py                            # opens with best model
    python3 watch.py --model models/exp7_latestart_cnnpolicy.zip
"""

import argparse
import os
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np

try:
    from PIL import Image, ImageTk, ImageDraw, ImageFont
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
ENV_ID     = "ALE/Tennis-v5"

ROOT_DIR   = os.path.dirname(SAGE_DIR)

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#0a1628"
BG2      = "#0f2040"
ACCENT   = "#00c9ff"
ACCENT2  = "#00b894"
FG       = "#e0f7fa"
FG_DIM   = "#607d8b"
RED      = "#ef5350"
GOLD     = "#ffd700"
GREEN    = "#69f0ae"
PANEL_BG = "#071220"
HUMAN_COL = "#ff9800"

# ── Tennis action space (18 discrete actions) ─────────────────────────────────
#  0:NOOP  1:FIRE  2:UP  3:RIGHT  4:LEFT  5:DOWN
#  6:UPRIGHT  7:UPLEFT  8:DOWNRIGHT  9:DOWNLEFT
# 10:UPFIRE 11:RIGHTFIRE 12:LEFTFIRE 13:DOWNFIRE
# 14:UPRIGHTFIRE 15:UPLEFTFIRE 16:DOWNRIGHTFIRE 17:DOWNLEFTFIRE

ACTION_NAMES = [
    "NOOP","FIRE","UP","RIGHT","LEFT","DOWN",
    "UPRIGHT","UPLEFT","DOWNRIGHT","DOWNLEFT",
    "UP+SWING","RIGHT+SWING","LEFT+SWING","DOWN+SWING",
    "UPRIGHT+SWING","UPLEFT+SWING","DOWNRIGHT+SWING","DOWNLEFT+SWING",
]

# Keyboard state → action mapping
# Keys tracked: up, down, left, right, fire(space)
_KEY_TO_ACTION = {
    # (up, down, left, right, fire)
    (False, False, False, False, False): 0,   # NOOP
    (False, False, False, False, True):  1,   # FIRE
    (True,  False, False, False, False): 2,   # UP
    (False, False, False, True,  False): 3,   # RIGHT
    (False, False, True,  False, False): 4,   # LEFT
    (False, True,  False, False, False): 5,   # DOWN
    (True,  False, False, True,  False): 6,   # UPRIGHT
    (True,  False, True,  False, False): 7,   # UPLEFT
    (False, True,  False, True,  False): 8,   # DOWNRIGHT
    (False, True,  True,  False, False): 9,   # DOWNLEFT
    (True,  False, False, False, True):  10,  # UP+SWING
    (False, False, False, True,  True):  11,  # RIGHT+SWING
    (False, False, True,  False, True):  12,  # LEFT+SWING
    (False, True,  False, False, True):  13,  # DOWN+SWING
    (True,  False, False, True,  True):  14,  # UPRIGHT+SWING
    (True,  False, True,  False, True):  15,  # UPLEFT+SWING
    (False, True,  False, True,  True):  16,  # DOWNRIGHT+SWING
    (False, True,  True,  False, True):  17,  # DOWNLEFT+SWING
}

# Modes
MODE_AI    = "Watch AI"
MODE_HUMAN = "Play Human"
MODE_HINT  = "AI Hint"

EXPERIMENT_NOTES = {
    "exp1_cnn_baseline":          "Baseline  lr=2.5e-4  buf=50k",
    "exp2_largebuffer":           "Large Buffer 100k  ★ BEST mean",
    "exp3_smallbuffer":           "Small Buffer 10k  recency-biased",
    "exp4_frequenttargetupdate":  "Target update every 250 steps",
    "exp5_raretargetupdate":      "Target update every 5000 steps",
    "exp6_hightrainfreq":         "Train every step  train_freq=1",
    "exp7_latestart":             "Late start 25k  ★ max +1.0 ever!",
    "exp8_conservativeexplore":   "Conservative ε_end=0.15 decay=0.40",
    "exp9_fastconverge":          "Fast converge  lr=5e-4 batch=64",
    "exp10_bestcombo":            "Best-combo  lr=2e-4 buf=75k",
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
    entries = []
    if os.path.exists(BEST_MODEL):
        entries.append(("★ Best Model  (dqn_model.zip)", BEST_MODEL))
    if os.path.isdir(MODELS_DIR):
        for fname in sorted(os.listdir(MODELS_DIR)):
            if fname.endswith(".zip"):
                key   = fname.replace("_cnnpolicy.zip", "").replace("sage_", "")
                note  = EXPERIMENT_NOTES.get(key, "")
                label = fname.replace("_cnnpolicy.zip", "").replace("_", " ").title()
                if note:
                    label = f"{label}  —  {note}"
                entries.append((label, os.path.join(MODELS_DIR, fname)))
                
    # ── Other Students' Models ──
    for student in ["kariza", "orpheus", "Emmanuel"]:
        student_dir = os.path.join(ROOT_DIR, student)
        student_models_dir = os.path.join(student_dir, "models")
        prefix = f"[{student.capitalize()}]"
        
        best = os.path.join(student_dir, "dqn_best.zip")
        if os.path.exists(best):
            entries.append((f"{prefix} ★ Best Model", best))
            
        latest = os.path.join(student_dir, "dqn_latest.zip")
        if os.path.exists(latest):
            entries.append((f"{prefix} Latest Model", latest))

        if os.path.isdir(student_models_dir):
            for fname in sorted(os.listdir(student_models_dir)):
                if fname.endswith(".zip"):
                    label = f"{prefix} {fname.replace('_cnnpolicy.zip', '').replace('_', ' ').title()}"
                    entries.append((label, os.path.join(student_models_dir, fname)))
                
    return entries


def make_render_env() -> VecFrameStack:
    env = make_atari_env(
        ENV_ID, n_envs=1, seed=0,
        vec_env_cls=DummyVecEnv,
        env_kwargs={"render_mode": "rgb_array"},
    )
    return VecFrameStack(env, n_stack=4)


def get_frame(env: VecFrameStack) -> np.ndarray | None:
    try:
        frames = env.render()
        if frames is None:
            return None
        if isinstance(frames, (list, tuple)):
            frames = frames[0]
        if isinstance(frames, np.ndarray):
            if frames.ndim == 4:
                frames = frames[0]
            return frames.astype(np.uint8)
    except Exception:
        pass
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


# ── Main App ──────────────────────────────────────────────────────────────────
class TennisWatcher:
    DISPLAY_W = 720
    DISPLAY_H = 540
    PANEL_W   = 380

    def __init__(self, root: tk.Tk, initial_model: str, max_episodes: int = 0):
        self.root         = root
        self.max_episodes = max_episodes

        self.env:   VecFrameStack | None = None
        self.model: DQN | None           = None
        self._model_path = ""

        self._running  = False
        self._paused   = False
        self._thread:  threading.Thread | None = None
        self._stop_evt = threading.Event()

        self.episode     = 0
        self.ep_reward   = 0.0
        self.ep_step     = 0
        self.all_rewards: list[float] = []
        self._speed_ms   = 50

        # play mode
        self._mode = MODE_AI

        # keyboard state for human play
        self._keys = {"up": False, "down": False,
                      "left": False, "right": False, "fire": False}
        self._hint_action = 0   # DQN's suggestion in HINT mode

        self._build_ui()
        self._bind_keys()
        self._populate_model_list()

        models = discover_models()
        initial_idx = 0
        for i, (_, path) in enumerate(models):
            if os.path.abspath(path) == os.path.abspath(initial_model):
                initial_idx = i
                break
        self.model_var.set(models[initial_idx][0] if models else "")
        self._on_model_change()

    # ── Keyboard bindings ──────────────────────────────────────────────────────
    def _bind_keys(self):
        self.root.focus_set()
        for key, sym in [
            ("up",    ["Up",    "w", "W"]),
            ("down",  ["Down",  "s", "S"]),
            ("left",  ["Left",  "a", "A"]),
            ("right", ["Right", "d", "D"]),
            ("fire",  ["space"]),
        ]:
            for s in sym:
                self.root.bind(f"<KeyPress-{s}>",
                               lambda e, k=key: self._key_press(k))
                self.root.bind(f"<KeyRelease-{s}>",
                               lambda e, k=key: self._key_release(k))

        # Pause on Escape / P
        self.root.bind("<Escape>", lambda e: self._toggle_pause())
        self.root.bind("<p>",      lambda e: self._toggle_pause())
        self.root.bind("<P>",      lambda e: self._toggle_pause())

    def _key_press(self, key: str):
        self._keys[key] = True
        self._update_key_display()

    def _key_release(self, key: str):
        self._keys[key] = False
        self._update_key_display()

    def _human_action(self) -> int:
        k = self._keys
        combo = (k["up"], k["down"], k["left"], k["right"], k["fire"])
        return _KEY_TO_ACTION.get(combo, 0)

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.root.title("Sage · DQN Tennis · Watch / Play")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        # Title
        tk.Label(
            self.root,
            text="SAGE  ·  DQN Tennis  ·  CnnPolicy  ·  ALE/Tennis-v5",
            bg=BG, fg=ACCENT, font=("Courier New", 13, "bold"), pady=6,
        ).pack(fill=tk.X)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # ── Left: canvas ──────────────────────────────────────────────────────
        left = tk.Frame(body, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH)

        self.canvas = tk.Canvas(
            left, width=self.DISPLAY_W, height=self.DISPLAY_H,
            bg="#000010", highlightthickness=2, highlightbackground=ACCENT,
        )
        self.canvas.pack()
        self._blank_canvas("Loading model…")

        # ── Keyboard diagram under canvas ─────────────────────────────────────
        kb = tk.Frame(left, bg=BG2, pady=6, padx=10)
        kb.pack(fill=tk.X, pady=(4, 0))

        # Title row
        tk.Label(kb, text="KEYBOARD CONTROLS", bg=BG2, fg=ACCENT,
                 font=("Courier New", 10, "bold")).pack()

        keys_frame = tk.Frame(kb, bg=BG2)
        keys_frame.pack(pady=(4, 2))

        def key_box(parent, text, w=5, h=2, color=FG, bg_col=BG):
            return tk.Label(parent, text=text, width=w, height=h,
                            bg=bg_col, fg=color, relief=tk.RAISED,
                            font=("Courier New", 10, "bold"), bd=2)

        # Row 1: W / Up
        row0 = tk.Frame(keys_frame, bg=BG2)
        row0.pack()
        tk.Label(row0, text="", width=5, bg=BG2).pack(side=tk.LEFT)
        key_box(row0, "W\n↑", color=HUMAN_COL).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Label(row0, text="   ", bg=BG2).pack(side=tk.LEFT)
        key_box(row0, "↑", color=HUMAN_COL).pack(side=tk.LEFT, padx=2, pady=2)

        # Row 2: A S D / Left Down Right
        row1 = tk.Frame(keys_frame, bg=BG2)
        row1.pack()
        key_box(row1, "A\n←", color=HUMAN_COL).pack(side=tk.LEFT, padx=2, pady=2)
        key_box(row1, "S\n↓", color=HUMAN_COL).pack(side=tk.LEFT, padx=2, pady=2)
        key_box(row1, "D\n→", color=HUMAN_COL).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Label(row1, text=" ", bg=BG2).pack(side=tk.LEFT)
        key_box(row1, "←", color=HUMAN_COL).pack(side=tk.LEFT, padx=2, pady=2)
        key_box(row1, "↓", color=HUMAN_COL).pack(side=tk.LEFT, padx=2, pady=2)
        key_box(row1, "→", color=HUMAN_COL).pack(side=tk.LEFT, padx=2, pady=2)

        # Space bar
        row2 = tk.Frame(keys_frame, bg=BG2)
        row2.pack()
        key_box(row2, "  SPACE = SWING / HIT  ", w=30, h=1,
                color=GOLD, bg_col="#1a2a00").pack(padx=2, pady=4)

        # Legend text
        tips = tk.Frame(kb, bg=BG2)
        tips.pack(fill=tk.X, pady=(2, 2))
        for tip, col in [
            ("Move + Space at same time = move AND swing", HUMAN_COL),
            ("Esc / P = Pause        Human mode = YOU play", FG_DIM),
        ]:
            tk.Label(tips, text=tip, bg=BG2, fg=col,
                     font=("Courier New", 9)).pack()

        # ── Right: panel ──────────────────────────────────────────────────────
        right = tk.Frame(body, bg=PANEL_BG, width=self.PANEL_W, padx=12, pady=10)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right.pack_propagate(False)
        self._build_panel(right)

    def _blank_canvas(self, msg=""):
        self.canvas.delete("all")
        self.canvas.create_rectangle(
            0, 0, self.DISPLAY_W, self.DISPLAY_H, fill="#000010", outline="")
        if msg:
            self.canvas.create_text(
                self.DISPLAY_W // 2, self.DISPLAY_H // 2,
                text=msg, fill=FG_DIM, font=("Courier New", 14))

    def _build_panel(self, p):
        def section(txt):
            tk.Label(p, text=txt, bg=PANEL_BG, fg=ACCENT,
                     font=("Courier New", 10, "bold")).pack(anchor=tk.W, pady=(10, 2))
            tk.Frame(p, bg=ACCENT, height=1).pack(fill=tk.X)

        def stat(label, var, color=FG):
            row = tk.Frame(p, bg=PANEL_BG)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=f"{label:<14}", bg=PANEL_BG,
                     fg=FG_DIM, font=("Courier New", 10)).pack(side=tk.LEFT)
            tk.Label(row, textvariable=var, bg=PANEL_BG,
                     fg=color, font=("Courier New", 11, "bold")).pack(side=tk.LEFT)

        btn = dict(bg=BG2, fg=FG, font=("Courier New", 11, "bold"),
                   relief=tk.FLAT, cursor="hand2", pady=6, padx=8,
                   activebackground=ACCENT, activeforeground=BG)

        # ── Mode selector ─────────────────────────────────────────────────────
        section("PLAY MODE")
        self.v_mode = tk.StringVar(value=MODE_AI)
        mode_row = tk.Frame(p, bg=PANEL_BG)
        mode_row.pack(fill=tk.X, pady=(6, 0))
        for mode, col in [(MODE_AI, ACCENT), (MODE_HUMAN, HUMAN_COL), (MODE_HINT, GREEN)]:
            tk.Radiobutton(
                mode_row, text=mode, variable=self.v_mode, value=mode,
                bg=PANEL_BG, fg=col, selectcolor=BG2,
                activebackground=PANEL_BG, activeforeground=col,
                font=("Courier New", 11, "bold"),
                command=self._on_mode_change,
            ).pack(side=tk.LEFT, padx=4)

        # ── Model selector ────────────────────────────────────────────────────
        section("DQN MODEL")
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            p, textvariable=self.model_var, state="readonly",
            font=("Courier New", 7), width=34)
        self.model_combo.pack(fill=tk.X, pady=(4, 0))
        self.model_combo.bind("<<ComboboxSelected>>",
                              lambda _: self._on_model_change())
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
        stat("Episode",   self.v_episode,  ACCENT)
        stat("Ep Reward", self.v_reward,   GOLD)
        stat("Ep Steps",  self.v_step,     FG)
        stat("Best Ep",   self.v_best_ep,  ACCENT2)
        stat("Mean All",  self.v_mean_all, FG)
        stat("Status",    self.v_status,   ACCENT2)

        # ── DQN hint (visible in HINT mode) ───────────────────────────────────
        section("DQN HINT")
        self.v_hint_action = tk.StringVar(value="—")
        self.v_hint_label  = tk.StringVar(value="—")
        stat("DQN says",  self.v_hint_action, GREEN)
        stat("My action", self.v_hint_label,  HUMAN_COL)

        # ── Keyboard indicator ────────────────────────────────────────────────
        section("KEYBOARD")
        self.v_keys = tk.StringVar(value="—")
        tk.Label(p, textvariable=self.v_keys, bg=PANEL_BG,
                 fg=HUMAN_COL, font=("Courier New", 13, "bold")).pack(anchor=tk.W, pady=2)

        # ── Training results ──────────────────────────────────────────────────
        section("TRAINING RESULTS")
        self.v_train_mean   = tk.StringVar(value="—")
        self.v_train_max    = tk.StringVar(value="—")
        self.v_train_last20 = tk.StringVar(value="—")
        stat("Train mean",  self.v_train_mean,   FG)
        stat("Train max",   self.v_train_max,    ACCENT2)
        stat("Last-20 avg", self.v_train_last20, FG)

        # ── Speed slider ──────────────────────────────────────────────────────
        section("SPEED")
        sp = tk.Frame(p, bg=PANEL_BG)
        sp.pack(fill=tk.X, pady=(4, 0))
        tk.Label(sp, text="Slow", bg=PANEL_BG,
                 fg=FG_DIM, font=("Courier New", 8)).pack(side=tk.LEFT)
        self.speed_slider = tk.Scale(
            sp, from_=150, to=10, orient=tk.HORIZONTAL,
            bg=PANEL_BG, fg=FG, troughcolor=BG2,
            highlightthickness=0, showvalue=False,
            command=lambda v: setattr(self, "_speed_ms", int(v)),
        )
        self.speed_slider.set(50)
        self.speed_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(sp, text="Fast", bg=PANEL_BG,
                 fg=FG_DIM, font=("Courier New", 8)).pack(side=tk.LEFT)

        # ── Buttons ───────────────────────────────────────────────────────────
        section("CONTROLS  (Esc / P = pause)")
        r1 = tk.Frame(p, bg=PANEL_BG)
        r1.pack(fill=tk.X, pady=(6, 2))
        self.btn_play = tk.Button(r1, text="▶  PLAY", **btn,
                                  command=self._toggle_pause)
        self.btn_play.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        tk.Button(r1, text="↺  NEW EP", **btn,
                  command=self._new_episode).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(p, text="■  STOP", command=self._stop,
                  **{**btn, "fg": RED}).pack(fill=tk.X, pady=(2, 0))

        # footer
        tk.Frame(p, bg=PANEL_BG).pack(fill=tk.Y, expand=True)
        tk.Label(p, text="n_envs=1  ·  200k steps  ·  CnnPolicy",
                 bg=PANEL_BG, fg=FG_DIM, font=("Courier New", 9)).pack(pady=(4, 0))

    # ── Key display ────────────────────────────────────────────────────────────
    def _update_key_display(self):
        parts = []
        if self._keys["up"]:    parts.append("↑")
        if self._keys["down"]:  parts.append("↓")
        if self._keys["left"]:  parts.append("←")
        if self._keys["right"]: parts.append("→")
        if self._keys["fire"]:  parts.append("SWING")
        self.v_keys.set("  ".join(parts) if parts else "—  (no key pressed)")

        if self._mode in (MODE_HUMAN, MODE_HINT):
            a = self._human_action()
            self.v_hint_label.set(ACTION_NAMES[a])

    # ── Model management ───────────────────────────────────────────────────────
    def _populate_model_list(self):
        self._models = discover_models()
        self.model_combo["values"] = [l for l, _ in self._models]

    def _on_model_change(self, *_):
        label = self.model_var.get()
        path  = next((p for l, p in self._models if l == label), None)
        if path and path != self._model_path:
            self._stop()
            self.v_status.set("Loading…")
            self._blank_canvas("Loading model…")
            threading.Thread(target=self._load_and_start,
                             args=(path,), daemon=True).start()

    def _on_mode_change(self):
        self._mode = self.v_mode.get()
        # update title colour based on mode
        col = {MODE_AI: ACCENT, MODE_HUMAN: HUMAN_COL, MODE_HINT: GREEN}.get(self._mode, ACCENT)
        self.canvas.config(highlightbackground=col)
        if self._mode == MODE_HUMAN:
            self.v_status.set("Human mode — use keys!")
        elif self._mode == MODE_HINT:
            self.v_status.set("Hint mode — DQN advises")
        else:
            self.v_status.set("AI mode")

    def _load_and_start(self, path: str):
        try:
            if self.env is not None:
                try:
                    self.env.close()
                except Exception:
                    pass
            self.env   = make_render_env()
            self.model = DQN.load(path, env=self.env)
            self._model_path = path

            key = os.path.basename(path).replace("_cnnpolicy.zip","").replace("sage_","")
            res = RESULTS.get(key, {})
            def _upd():
                self.v_train_mean.set(f"{res.get('mean','—'):.2f}" if res else "—")
                self.v_train_max.set(f"{res.get('max','—'):.2f}"   if res else "—")
                self.v_train_last20.set(f"{res.get('last20','—'):.2f}" if res else "—")
            self.root.after(0, _upd)
            self.root.after(0, lambda: self.v_status.set("Ready"))
            self._start_episode()
        except Exception as ex:
            self.root.after(0, lambda: self.v_status.set(f"Error: {ex}"))

    # ── Game loop ──────────────────────────────────────────────────────────────
    def _start_episode(self):
        self._stop_evt.clear()
        self._paused  = False
        self._running = True
        self.root.after(0, lambda: self.btn_play.config(text="⏸  PAUSE"))
        self._thread  = threading.Thread(target=self._game_loop, daemon=True)
        self._thread.start()

    def _game_loop(self):
        if self.env is None or self.model is None:
            return

        obs = self.env.reset()
        self.episode   += 1
        self.ep_reward  = 0.0
        self.ep_step    = 0
        self.root.after(0, self._refresh_stats)

        while not self._stop_evt.is_set():
            # pause
            while self._paused and not self._stop_evt.is_set():
                time.sleep(0.05)
            if self._stop_evt.is_set():
                break

            mode = self._mode

            # decide action
            if mode == MODE_AI:
                action, _ = self.model.predict(obs, deterministic=True)
                self._hint_action = int(action[0])

            elif mode == MODE_HUMAN:
                human_a = self._human_action()
                action  = np.array([human_a])
                self._hint_action = human_a

            else:  # HINT
                human_a = self._human_action()
                ai_a, _ = self.model.predict(obs, deterministic=True)
                self._hint_action = int(ai_a[0])
                action = np.array([human_a])   # human drives, AI advises

            obs, reward, done, info = self.env.step(action)

            self.ep_reward  += float(reward[0])
            self.ep_step    += 1

            # render
            frame = get_frame(self.env)
            if frame is not None:
                overlay_mode = mode
                hint_a       = self._hint_action
                human_a_now  = self._human_action()
                self.root.after(0, lambda f=frame, m=overlay_mode,
                                          ha=hint_a, ua=human_a_now:
                                self._display_frame(f, m, ha, ua))

            # update hints in panel
            if mode in (MODE_HINT,):
                self.root.after(0, lambda a=self._hint_action:
                                self.v_hint_action.set(ACTION_NAMES[a]))

            self.root.after(0, self._refresh_stats)
            time.sleep(self._speed_ms / 1000.0)

            if done[0]:
                self.all_rewards.append(self.ep_reward)
                if self.max_episodes and self.episode >= self.max_episodes:
                    self.root.after(0, lambda: self.v_status.set("Done"))
                    return
                time.sleep(0.6)
                obs = self.env.reset()
                self.episode   += 1
                self.ep_reward  = 0.0
                self.ep_step    = 0
                self.root.after(0, self._refresh_stats)

        self._running = False
        self.root.after(0, lambda: self.btn_play.config(text="▶  PLAY"))

    # ── Frame display (with optional overlay) ─────────────────────────────────
    def _display_frame(self, frame: np.ndarray,
                       mode: str, hint_action: int, human_action: int):
        if not PIL_OK:
            return
        try:
            img = Image.fromarray(frame)
            img = img.resize((self.DISPLAY_W, self.DISPLAY_H), Image.NEAREST)

            # draw overlay text on the frame image
            draw = ImageDraw.Draw(img)

            if mode == MODE_AI:
                label = f"AI: {ACTION_NAMES[hint_action]}"
                draw.rectangle([4, 4, len(label)*7+8, 20], fill=(0, 40, 80, 180))
                draw.text((6, 5), label, fill=(0, 201, 255))

            elif mode == MODE_HUMAN:
                label = f"YOU: {ACTION_NAMES[human_action]}"
                draw.rectangle([4, 4, len(label)*7+8, 20], fill=(50, 30, 0, 180))
                draw.text((6, 5), label, fill=(255, 152, 0))

            elif mode == MODE_HINT:
                you_lbl = f"YOU: {ACTION_NAMES[human_action]}"
                ai_lbl  = f"DQN: {ACTION_NAMES[hint_action]}"
                draw.rectangle([4, 4, max(len(you_lbl), len(ai_lbl))*7+8, 32],
                               fill=(0, 20, 10, 200))
                draw.text((6, 5),  you_lbl, fill=(255, 152, 0))
                draw.text((6, 18), ai_lbl,  fill=(105, 240, 174))

            photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas._photo_ref = photo
        except Exception:
            pass

    def _refresh_stats(self):
        self.v_episode.set(str(self.episode))
        self.v_reward.set(f"{self.ep_reward:.1f}")
        self.v_step.set(f"{self.ep_step:,}")
        if self.all_rewards:
            self.v_best_ep.set(f"{max(self.all_rewards):.1f}")
            self.v_mean_all.set(f"{np.mean(self.all_rewards):.2f}")
        if not self._paused and self._running:
            col = {MODE_AI: "AI playing", MODE_HUMAN: "You playing!",
                   MODE_HINT: "You + DQN hint"}.get(self._mode, "Running")
            self.v_status.set(col)

    # ── Controls ──────────────────────────────────────────────────────────────
    def _toggle_pause(self):
        if not self._running:
            self._start_episode()
            return
        self._paused = not self._paused
        self.btn_play.config(text="▶  PLAY" if self._paused else "⏸  PAUSE")
        self.v_status.set("Paused  (P to resume)" if self._paused
                          else self._mode)

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
        print("\n  Missing Pillow:\n  pip install Pillow\n")
        return

    p = argparse.ArgumentParser(description="Sage · DQN Tennis Watch/Play")
    p.add_argument("--model",    default=BEST_MODEL)
    p.add_argument("--episodes", type=int, default=0)
    args = p.parse_args()

    if not os.path.exists(args.model):
        found = discover_models()
        if not found:
            print("  No models found — run train.py --all first.")
            return
        args.model = found[0][1]

    root = tk.Tk()
    app  = TennisWatcher(root, initial_model=args.model,
                         max_episodes=args.episodes)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
