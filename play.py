"""
play.py — ALE/Tennis-v5  |  Student: Kariza  |  Policy: CnnPolicy
────────────────────────────────────────────────────────────────────
Evaluation GUI with experiment selector.
Auto-discovers all trained models from logs/ and models/ folders.
Switch between experiments live — comparison chart updates after each episode.

Usage:
    python3 play.py
    python3 play.py --model dqn_best.zip
"""

import argparse
import glob
import os
import threading
import time
import queue

import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import ttk

import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

gym.register_envs(ale_py)

ENV_ID       = "ALE/Tennis-v5"
ACTION_NAMES = [
    "NOOP","FIRE","UP","RIGHT","LEFT","DOWN",
    "UPRIGHT","UPLEFT","DOWNRIGHT","DOWNLEFT",
    "UPFIRE","RIGHTFIRE","LEFTFIRE","DOWNFIRE",
    "UPRIGHTFIRE","UPLEFTFIRE","DOWNRIGHTFIRE","DOWNLEFTFIRE",
]

# ── Kariza palette — rose & lavender ──────────────────────────────────────
BG    = "#1a0a14"   # deep wine
PANEL = "#2b1020"   # dark rose
CARD  = "#33152a"   # muted magenta card
ACC1  = "#ff85c0"   # soft pink  — primary
ACC2  = "#c94b8a"   # deep rose  — active/stop
ACC3  = "#b57bee"   # lavender   — secondary
ACC4  = "#ffb3d9"   # blush      — highlight/best
TEXT  = "#ffe6f2"   # soft white-pink
MUTED = "#8a4a6a"   # dusty rose muted
DARK  = "#120008"   # near-black with red tint
FT    = ("Georgia", 8)
FB    = ("Georgia", 9, "bold")

EXP_COLORS = [
    "#ff85c0","#c94b8a","#b57bee","#ffb3d9","#f472b6",
    "#e879f9","#fb7185","#c084fc","#f9a8d4","#a78bfa",
]


# ──────────────────────────────────────────────────────────────────────────────
# Model discovery
# ──────────────────────────────────────────────────────────────────────────────
def discover_models() -> list[dict]:
    models = []
    for fname, label in [
        ("dqn_best.zip",   "★ BEST  (dqn_best.zip)"),
        ("dqn_latest.zip", "Latest (dqn_latest.zip)"),
        ("dqn_model.zip",  "Final  (dqn_model.zip)"),
    ]:
        if os.path.exists(fname):
            models.append({"label": label, "path": fname, "exp_id": None})

    # models/ folder — assignment spec naming
    for path in sorted(glob.glob("models/exp*.zip")):
        fname = os.path.basename(path).replace(".zip","")
        label = fname.replace("_", " ").replace("cnnpolicy","CNN")
        models.append({"label": label, "path": path, "exp_id": None})

    # kariza/models/ folder — group member models
    for path in sorted(glob.glob("kariza/models/exp*.zip")):
        fname = os.path.basename(path).replace(".zip","")
        label = f"★ KARIZA: {fname.replace('_', ' ').replace('cnnpolicy','CNN')}"
        models.append({"label": label, "path": path, "exp_id": None})

    # logs/ folder
    for path in sorted(glob.glob("logs/experiments_*/exp_*_model.zip")):
        fname = os.path.basename(path)
        try:
            exp_num = int(fname.split("_")[1])
        except (IndexError, ValueError):
            exp_num = 0
        label_raw = fname.replace("_model.zip","").replace("_"," ")
        label = f"Exp {exp_num:>2} — {' '.join(label_raw.split()[2:])}"
        models.append({"label": label, "path": path, "exp_id": exp_num})

    seen   = set()
    unique = []
    for m in models:
        if m["path"] not in seen:
            seen.add(m["path"])
            unique.append(m)

    return unique if unique else [
        {"label":"No models found — run experiments first",
         "path": None, "exp_id": None}
    ]


# ──────────────────────────────────────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────────────────────────────────────
class PlayGUI:
    def __init__(self, root: tk.Tk, args):
        self.root    = root
        self.args    = args
        self.running = False
        self._fq: queue.Queue = queue.Queue(maxsize=4)

        self.current_label = ""
        self.ep_rewards    = []
        self.play_ep       = 0
        self.total_steps   = 0
        self.comparison: dict[str, list[float]] = {}

        self._switch_model_path  = None
        self._switch_model_label = None
        self._switch_lock        = threading.Lock()

        root.title("✦  Kariza's Tennis Lab  ·  DQN Experiment Player  ✦")
        root.configure(bg=BG)
        root.geometry("1380x880")
        self._build()
        root.after(28, self._poll_frame)

    # ── build ─────────────────────────────────────────────────────────────────
    def _build(self):
        # header
        hdr = tk.Frame(self.root, bg=PANEL, height=52)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  ✦ KARIZA'S TENNIS LAB  ✦  Experiment Player",
                 font=("Georgia",14,"bold"), bg=PANEL, fg=ACC1
                 ).pack(side=tk.LEFT, padx=20, pady=10)
        tk.Label(hdr,
                 text="ALE/Tennis-v5  ·  CnnPolicy  ·  Switch experiments live ✦",
                 font=FT, bg=PANEL, fg=MUTED).pack(side=tk.LEFT, padx=10)
        tk.Label(hdr, text="GreedyQPolicy  ✦  argmax Q(s,a)",
                 font=FT, bg=PANEL, fg=ACC4).pack(side=tk.RIGHT, padx=20)

        # selector bar
        sel = tk.Frame(self.root, bg=CARD, height=46)
        sel.pack(fill=tk.X)
        sel.pack_propagate(False)

        tk.Label(sel, text="  ✦ Model:", font=("Georgia",9,"bold"), bg=CARD, fg=ACC1
                 ).pack(side=tk.LEFT, padx=(12,4), pady=10)

        self.models       = discover_models()
        self.model_labels = [m["label"] for m in self.models]
        self.sel_var      = tk.StringVar(value=self.model_labels[0])

        sty = ttk.Style()
        sty.configure("SEL.TCombobox",
                      fieldbackground="white", background="white",
                      foreground="black", arrowcolor="black",
                      selectbackground="#cccccc", selectforeground="black",
                      font=("Courier",9,"bold"))
        sty.map("SEL.TCombobox",
                fieldbackground=[("readonly","white")],
                foreground=[("readonly","black")])

        self.sel_combo = ttk.Combobox(
            sel, textvariable=self.sel_var,
            values=self.model_labels, state="readonly",
            style="SEL.TCombobox", width=52,
            font=("Courier",9,"bold"))
        self.sel_combo.pack(side=tk.LEFT, padx=6, pady=10)

        for text, cmd in [
            (" ▶ LOAD & PLAY ", self._load_selected),
            (" ↺ REFRESH ",     self._refresh_models),
        ]:
            tk.Button(sel, text=text, bg="white", fg="black",
                      font=FB, relief=tk.FLAT, padx=8, cursor="hand2",
                      activebackground="#dddddd", activeforeground="black",
                      command=cmd).pack(side=tk.LEFT, padx=3)

        self.run_btn = tk.Button(
            sel, text=" STOP ", bg=ACC2, fg="black",
            font=FB, relief=tk.FLAT, padx=10, cursor="hand2",
            activebackground=ACC2, activeforeground="black",
            command=self._toggle)
        self.run_btn.pack(side=tk.LEFT, padx=3)

        tk.Button(sel, text=" RESET SCORES ", bg="white", fg="black",
                  font=FB, relief=tk.FLAT, padx=8, cursor="hand2",
                  activebackground="#dddddd", activeforeground="black",
                  command=self._reset_scores).pack(side=tk.LEFT, padx=3)

        self.cur_lbl = tk.Label(sel, text="No model loaded",
                                font=FT, bg=CARD, fg=MUTED)
        self.cur_lbl.pack(side=tk.RIGHT, padx=16)

        # body
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # LEFT — game feed
        left = tk.Frame(body, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        stat_f = tk.Frame(left, bg=CARD, height=36)
        stat_f.pack(fill=tk.X, pady=(0,4))
        stat_f.pack_propagate(False)
        self.sv = {}
        for lbl, key, col in [
            ("EPISODE","ep",ACC1), ("SCORE","rew",ACC2),
            ("BEST","best",ACC4), ("STEPS","steps",TEXT),
            ("ACTION","act",ACC3),
        ]:
            f = tk.Frame(stat_f, bg=CARD, padx=10)
            f.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=4)
            tk.Label(f, text=lbl, font=("Georgia",7,"bold"),
                     bg=CARD, fg=MUTED).pack(anchor=tk.W)
            var = tk.StringVar(value="--")
            tk.Label(f, textvariable=var,
                     font=("Georgia",11,"bold"), bg=CARD, fg=col
                     ).pack(anchor=tk.W)
            self.sv[key] = var

        cf = tk.Frame(left, bg=CARD)
        cf.pack(fill=tk.BOTH, expand=True, pady=(0,4))
        tk.Label(cf, text="✦  LIVE FEED  ·  Tennis-v5  ·  210×160 RGB",
                 font=("Georgia",8,"bold"), bg=CARD, fg=ACC1).pack(anchor=tk.W, padx=8, pady=(4,0))
        self.canvas = tk.Canvas(cf, bg=DARK, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0,6))
        self._ph()

        ec = tk.Frame(left, bg=CARD, height=170)
        ec.pack(fill=tk.X, pady=(0,4))
        ec.pack_propagate(False)
        tk.Label(ec, text="✦  EPISODE REWARDS  ·  Current Model",
                 font=("Georgia",8,"bold"), bg=CARD, fg=ACC1
                 ).pack(anchor=tk.W, padx=8, pady=(4,0))
        self.ep_fig, self.ep_ax = plt.subplots(figsize=(7,1.4), facecolor="#2b1020")
        self.ep_ax.set_facecolor("#1a0a14")
        self._sax(self.ep_ax)
        self.ep_fig.tight_layout(pad=0.4)
        self.ep_cv = FigureCanvasTkAgg(self.ep_fig, master=ec)
        self.ep_cv.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                        padx=6, pady=(0,6))

        self.status_var = tk.StringVar(
            value="✦  Select a model from the dropdown and press  ▶ LOAD & PLAY")
        tk.Label(left, textvariable=self.status_var,
                 font=FT, bg=BG, fg=MUTED, anchor=tk.W
                 ).pack(fill=tk.X, pady=2)

        tk.Frame(body, bg="#1a1a40", width=2
                 ).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # RIGHT — comparison
        right = tk.Frame(body, bg=BG, width=380)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        tk.Label(right, text="✦  EXPERIMENT COMPARISON",
                 font=("Georgia",9,"bold"), bg=BG, fg=ACC1
                 ).pack(anchor=tk.W, padx=8, pady=(0,4))
        tk.Label(right,
                 text="Solid = mean last 10 eps  ·  Faded = best score",
                 font=("Georgia",7), bg=BG, fg=MUTED, justify=tk.LEFT
                 ).pack(anchor=tk.W, padx=8)

        cmp_f = tk.Frame(right, bg=CARD)
        cmp_f.pack(fill=tk.BOTH, expand=True, pady=(6,4))
        self.cmp_fig, self.cmp_ax = plt.subplots(figsize=(4.2,6.5),
                                                   facecolor="#2b1020")
        self.cmp_ax.set_facecolor("#1a0a14")
        self._sax(self.cmp_ax)
        self.cmp_fig.subplots_adjust(
            left=0.38, right=0.97, top=0.97, bottom=0.06)
        self.cmp_cv = FigureCanvasTkAgg(self.cmp_fig, master=cmp_f)
        self.cmp_cv.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                         padx=4, pady=4)

        sb_f = tk.Frame(right, bg=CARD)
        sb_f.pack(fill=tk.X, pady=(0,4))
        tk.Label(sb_f, text="✦  SCOREBOARD  ·  mean last 10 eps",
                 font=("Georgia",7,"bold"), bg=CARD, fg=ACC1
                 ).pack(anchor=tk.W, padx=8, pady=(6,2))
        self.score_text = tk.Text(
            sb_f, height=12, bg=DARK, fg=TEXT,
            font=("Courier",8), relief=tk.FLAT,
            state=tk.DISABLED, bd=4)
        self.score_text.pack(fill=tk.X, padx=6, pady=(0,6))

    # ── helpers ───────────────────────────────────────────────────────────────
    def _ph(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            400, 200, fill=MUTED,
            font=("Courier",11), justify=tk.CENTER,
            text="✦  Select a model from the dropdown  ✦\n\n"
                 "Press  ▶ LOAD & PLAY  to begin\n\n"
                 "Comparison chart builds as you play each experiment")

    def _sax(self, ax):
        ax.tick_params(colors=MUTED, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1a1a40")

    def _label_color(self, label: str) -> str:
        return EXP_COLORS[hash(label) % len(EXP_COLORS)]

    def _push(self, frame):
        if self._fq.full():
            try: self._fq.get_nowait()
            except queue.Empty: pass
        self._fq.put(frame)

    def _poll_frame(self):
        try:
            frame = self._fq.get_nowait()
            w = max(self.canvas.winfo_width(),  1)
            h = max(self.canvas.winfo_height(), 1)
            img   = Image.fromarray(frame).resize((w, h), Image.NEAREST)
            photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(w//2, h//2, image=photo,
                                     anchor=tk.CENTER)
            self.canvas._ref = photo
        except queue.Empty:
            pass
        self.root.after(28, self._poll_frame)

    # ── charts ────────────────────────────────────────────────────────────────
    def _redraw_ep_chart(self):
        rewards = list(self.ep_rewards)
        self.ep_ax.clear()
        self.ep_ax.set_facecolor(DARK)
        self._sax(self.ep_ax)
        if rewards:
            xs    = list(range(1, len(rewards)+1))
            color = self._label_color(self.current_label)
            self.ep_ax.fill_between(xs, rewards, alpha=0.2, color=color)
            self.ep_ax.plot(xs, rewards, color=color, lw=1.3)
            if len(rewards) >= 5:
                w  = min(5, len(rewards))
                ma = np.convolve(rewards, np.ones(w)/w, mode="valid")
                self.ep_ax.plot(xs[w-1:], ma, color="white", lw=2)
            self.ep_ax.set_title(
                self.current_label[:50], color=TEXT, fontsize=7, pad=3)
        self.ep_fig.tight_layout(pad=0.4)
        self.ep_cv.draw()

    def _redraw_comparison(self):
        self.cmp_ax.clear()
        self.cmp_ax.set_facecolor(DARK)
        self._sax(self.cmp_ax)
        if not self.comparison:
            self.cmp_fig.subplots_adjust(
                left=0.38, right=0.97, top=0.97, bottom=0.06)
            self.cmp_cv.draw()
            return
        labels = list(self.comparison.keys())
        means  = [np.mean(v[-10:]) if v else 0
                  for v in self.comparison.values()]
        bests  = [max(v) if v else 0
                  for v in self.comparison.values()]
        colors = [self._label_color(l) for l in labels]
        y      = np.arange(len(labels))

        bars = self.cmp_ax.barh(y-0.2, means, height=0.35,
                                color=colors, alpha=0.9,
                                label="Mean last 10")
        self.cmp_ax.barh(y+0.2, bests, height=0.35,
                         color=colors, alpha=0.35,
                         label="Best episode")
        for bar, val in zip(bars, means):
            if val != 0:
                self.cmp_ax.text(
                    bar.get_width() + 0.1,
                    bar.get_y() + bar.get_height()/2,
                    f"{val:.2f}", va="center",
                    color=TEXT, fontsize=7)
        short = [l[:22] for l in labels]
        self.cmp_ax.set_yticks(y)
        self.cmp_ax.set_yticklabels(short, color=TEXT, fontsize=7)
        self.cmp_ax.set_xlabel("Reward", color=MUTED, fontsize=7)
        self.cmp_ax.legend(fontsize=6, facecolor=PANEL,
                           labelcolor=TEXT, loc="lower right")
        if means:
            best_idx = int(np.argmax(means))
            self.cmp_ax.get_yticklabels()[best_idx].set_color(ACC4)
            self.cmp_ax.get_yticklabels()[best_idx].set_fontweight("bold")
        self.cmp_fig.subplots_adjust(
            left=0.38, right=0.97, top=0.97, bottom=0.06)
        self.cmp_cv.draw()
        self._update_scoreboard(labels, means, bests)

    def _update_scoreboard(self, labels, means, bests):
        ranked = sorted(zip(labels, means, bests),
                        key=lambda x: x[1], reverse=True)
        lines = []
        for i, (lbl, mean, best) in enumerate(ranked, 1):
            marker = "★" if i == 1 else f"{i} "
            lines.append(f"{marker} {lbl[:24]:<24}\n"
                         f"   mean={mean:>7.2f}  best={best:>7.2f}\n")
        self.score_text.config(state=tk.NORMAL)
        self.score_text.delete("1.0", tk.END)
        self.score_text.insert(tk.END, "\n".join(lines))
        self.score_text.config(state=tk.DISABLED)

    # ── controls ──────────────────────────────────────────────────────────────
    def _refresh_models(self):
        self.models       = discover_models()
        self.model_labels = [m["label"] for m in self.models]
        self.sel_combo["values"] = self.model_labels
        self.status_var.set(
            f"Refreshed — {len(self.models)} models found.")

    def _load_selected(self):
        sel_label  = self.sel_var.get()
        model_info = next(
            (m for m in self.models if m["label"] == sel_label), None)
        if model_info is None or model_info["path"] is None:
            self.status_var.set("No valid model selected.")
            return
        path = model_info["path"]
        if not os.path.exists(path):
            self.status_var.set(f"File not found: {path}")
            return

        if self.running:
            with self._switch_lock:
                self._switch_model_path  = path
                self._switch_model_label = sel_label
            self.status_var.set(
                f"Switching to {sel_label} after this episode...")
        else:
            self.running       = True
            self.ep_rewards.clear()
            self.play_ep       = 0
            self.total_steps   = 0
            self.current_label = sel_label
            self.cur_lbl.config(text=sel_label, fg=ACC1)
            self.run_btn.config(text=" STOP ", bg=ACC2, fg="black",
                                activebackground=ACC2)
            self.status_var.set(f"Loading {sel_label}...")
            threading.Thread(
                target=self._play_thread,
                args=(path, sel_label),
                daemon=True
            ).start()

    def _toggle(self):
        if self.running:
            self.running = False
            self.run_btn.config(text=" START ", bg="white", fg="black",
                                font=FB, activebackground="#dddddd",
                                activeforeground="black")
            self.status_var.set("Stopped.")
        else:
            self._load_selected()

    def _reset_scores(self):
        self.running = False
        time.sleep(0.12)
        self.ep_rewards.clear()
        self.play_ep = self.total_steps = 0
        self.comparison.clear()
        for v in self.sv.values():
            v.set("--")
        for ax, cv, fig in [
            (self.ep_ax,  self.ep_cv,  self.ep_fig),
            (self.cmp_ax, self.cmp_cv, self.cmp_fig),
        ]:
            ax.clear(); ax.set_facecolor(DARK); self._sax(ax)
        self.ep_fig.tight_layout(pad=0.4); self.ep_cv.draw()
        self.cmp_fig.subplots_adjust(
            left=0.38, right=0.97, top=0.97, bottom=0.06)
        self.cmp_cv.draw()
        self.score_text.config(state=tk.NORMAL)
        self.score_text.delete("1.0", tk.END)
        self.score_text.config(state=tk.DISABLED)
        self._ph()
        self.cur_lbl.config(text="No model loaded", fg=MUTED)
        self.run_btn.config(text=" START ", bg="white", fg="black",
                            font=FB, activebackground="#dddddd",
                            activeforeground="black")
        self.status_var.set("Scores reset.")

    # ── play thread ───────────────────────────────────────────────────────────
    def _play_thread(self, initial_path: str, initial_label: str):
        try:
            current_path  = initial_path
            current_label = initial_label
            model = DQN.load(current_path)

            model_env = VecFrameStack(
                make_atari_env(ENV_ID, n_envs=1,
                               vec_env_cls=DummyVecEnv, seed=0),
                n_stack=4)
            display_env = gym.make(ENV_ID, render_mode="rgb_array")

            self.status_var.set(
                f"Playing: {current_label}  |  CnnPolicy  |  argmax Q(s,a)")

            while self.running:
                with self._switch_lock:
                    if self._switch_model_path is not None:
                        current_path  = self._switch_model_path
                        current_label = self._switch_model_label
                        self._switch_model_path  = None
                        self._switch_model_label = None
                        model = DQN.load(current_path)
                        self.current_label = current_label
                        self.ep_rewards.clear()
                        self.play_ep = 0
                        self.cur_lbl.config(text=current_label, fg=ACC1)
                        self.status_var.set(
                            f"Switched → {current_label}  |  CnnPolicy")

                obs = model_env.reset()
                display_env.reset()
                ep_r, done = 0.0, False

                while not done and self.running:
                    action, _ = model.predict(obs, deterministic=True)
                    action = np.atleast_1d(action)
                    obs, rew, dones, _ = model_env.step(action)
                    done   = bool(np.atleast_1d(dones)[0])
                    ep_r  += float(np.atleast_1d(rew)[0])
                    self.total_steps += 1

                    _, _, term, trunc, _ = display_env.step(int(action[0]))
                    frame = display_env.render()
                    if frame is not None:
                        self._push(frame)
                    if term or trunc:
                        display_env.reset()

                    act_name = ACTION_NAMES[int(action[0])] \
                               if int(action[0]) < len(ACTION_NAMES) \
                               else str(int(action[0]))
                    self.sv["act"].set(act_name)
                    self.sv["rew"].set(f"{ep_r:.2f}")
                    self.sv["steps"].set(f"{self.total_steps:,}")
                    time.sleep(0.016)

                if self.running:
                    self.play_ep += 1
                    self.ep_rewards.append(ep_r)
                    best = max(self.ep_rewards)
                    self.sv["ep"].set(str(self.play_ep))
                    self.sv["best"].set(f"{best:.2f}")

                    if current_label not in self.comparison:
                        self.comparison[current_label] = []
                    self.comparison[current_label].append(ep_r)

                    self.root.after(0, self._redraw_ep_chart)
                    self.root.after(0, self._redraw_comparison)

                    mean10 = np.mean(self.ep_rewards[-10:])
                    self.status_var.set(
                        f"{current_label[:38]}  |  "
                        f"ep={self.play_ep}  "
                        f"reward={ep_r:.2f}  "
                        f"best={best:.2f}  "
                        f"mean10={mean10:.2f}")

            model_env.close()
            display_env.close()
            self.running = False
            self.run_btn.config(text=" START ", bg="white", fg="black",
                                font=FB, activebackground="#dddddd",
                                activeforeground="black")

        except Exception as exc:
            self.status_var.set(f"Error: {exc}")
            self.running = False
            self.run_btn.config(text=" START ", bg="white", fg="black",
                                font=FB, activebackground="#dddddd",
                                activeforeground="black")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Play Tennis-v5 experiment models — Kariza")
    p.add_argument("--model", default=None)
    args = p.parse_args()

    root = tk.Tk()
    gui  = PlayGUI(root, args)

    if args.model and os.path.exists(args.model):
        if args.model not in [m["path"] for m in gui.models]:
            gui.models.append(
                {"label": args.model, "path": args.model, "exp_id": None})
            gui.model_labels.append(args.model)
            gui.sel_combo["values"] = gui.model_labels
        gui.sel_var.set(args.model)
        root.after(800, gui._load_selected)

    root.mainloop()
