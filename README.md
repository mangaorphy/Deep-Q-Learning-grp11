# Deep Q-Learning (DQN) - Atari Tennis Experiments

## Demo Video

Watch a demo of the trained agent playing Atari Tennis:

[![DQN Atari Tennis Demo](https://img.youtube.com/vi/vAPtx781yY0/0.jpg)](https://youtu.be/vAPtx781yY0)

Or view directly: https://youtu.be/vAPtx781yY0

## Overview

This project trains Deep Q-Network (DQN) agents to play Atari Tennis using both **CnnPolicy** and **MlpPolicy**, comparing their performance across different hyperparameter configurations.

## Why CNN-Based Models Outperform MLP Models


Our experiments consistently demonstrate that **CNN policies significantly outperform MLP policies** on the Atari Tennis task. Here's why:

### CNN (Convolutional Neural Network) Advantages

1. **Spatial Feature Extraction**
    - CNNs use convolutional filters to automatically detect spatial patterns in game frames
    - Learns hierarchical representations: edges → shapes → objects (paddle, ball, court)
    - Naturally captures local spatial relationships

2. **Efficiency for Image Data**
    - 84×84 grayscale frames = 7,056 pixels
    - CNN dramatically reduces parameters through weight sharing
    - MLP would need 7,056+ fully connected inputs (massive overfitting risk)
    - CNN filters are reusable across different regions of the image

3. **Better Generalization**
    - CNNs learn visual features relevant to the game (ball trajectory, paddle position)
    - Recognizes objects regardless of exact pixel location
    - MLP memorizes absolute pixel positions (brittle, doesn't generalize)

### MLP (Multi-Layer Perceptron) Limitations

- Treats each pixel as independent input → loses all spatial structure
- 7,056 inputs → massive parameter count → prone to overfitting
- Cannot capture relationships between nearby pixels
- Requires significantly more training data to achieve similar performance

### Experimental Results

**Exp2_LargeBuffer** (Large replay buffer experiment):
- **CNN Policy**: Episodes 56-67 show dramatic improvement (rewards: -1 to -17)
- **MLP Policy**: Plateaus at -24 (no significant learning)
- **Conclusion**: CNN learns effective strategies; MLP cannot extract meaningful patterns from raw pixels

**Recommendation**: Always use **CnnPolicy for Atari/image-based tasks**, MlpPolicy for low-dimensional state spaces (e.g., cart position, velocities).

## Quick Start

### Prerequisites

- Python 3.10+
- conda (Anaconda)

### Setup

```bash
# Create conda environment
conda create -n dqn python=3.10 -y

# Activate environment
conda activate dqn

# Install dependencies
pip install gymnasium==0.29.1 stable-baselines3==2.3.0 torch numpy matplotlib tqdm ale-py "numpy<2"
pip install "gymnasium[atari,accept-rom-license]" "gymnasium[other]"
```

### Run All Experiments

```bash
conda run -n dqn python train.py
```

This trains all 10 experiments with both CnnPolicy and MlpPolicy (20 total training runs).

**Estimated time:** 4-8 hours (depending on your machine)

## How to Add Your Own Experiments

### Step 1: Open `train.py`

Find the **`EXPERIMENTS`** list near the bottom of the file (around line 400):

```python
EXPERIMENTS = [
    {
        "name": "Exp1_AggressiveLR",
        "config_changes": {"learning_rate": 0.001, "gamma": 0.95, "batch_size": 64},
    },
    # ... more experiments
]
```

### Step 2: Add Your Experiment

Add a new dictionary to the list with your configuration:

```python
{
    "name": "YOUR_EXPERIMENT_NAME",
    "config_changes": {
        "learning_rate": 1e-4,      # Try different values
        "gamma": 0.99,               # Discount factor (0.9-0.9999)
        "batch_size": 32,            # Batch size (16, 32, 64, 128)
        "epsilon_end": 0.05,         # Final exploration rate
        "total_timesteps": 100000,   # Training timesteps
    },
},
```

### Step 3: Run Training

```bash
conda run -n dqn python train.py
```

## Configuration Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `learning_rate` | 1e-4 | 5e-6 to 1e-3 | How fast the model learns |
| `gamma` | 0.99 | 0.9 to 0.9999 | Discount factor (future reward weight) |
| `batch_size` | 32 | 16 to 128 | Samples per training step |
| `buffer_size` | 100000 | 50000+ | Experience replay buffer size |
| `epsilon_end` | 0.05 | 0.01 to 0.2 | Final exploration rate |
| `total_timesteps` | 100000 | 50000+ | Total environment steps to train |

## Output Files

After training completes:

- **`results_YYYYMMDD_HHMMSS.json`** - All experiment results with statistics
- **`models/`** folder - Saved DQN models (.zip files)
  - `exp1_cnnpolicy.zip`
  - `exp1_mlppolicy.zip`
  - etc.

### Results File Format

```json
[
  {
    "experiment": "Exp1_AggressiveLR",
    "policy": "CnnPolicy",
    "mean_reward": 2.5,
    "max_reward": 8.0,
    "episodes": 45,
    "time": 120.5
  },
  ...
]
```

## Example: Adding 3 New Experiments

```python
EXPERIMENTS = [
    # ... existing experiments ...
    {
        "name": "MyExp1_HighLR",
        "config_changes": {"learning_rate": 5e-4, "gamma": 0.99, "batch_size": 64},
    },
    {
        "name": "MyExp2_SmallBatch",
        "config_changes": {"learning_rate": 1e-4, "gamma": 0.99, "batch_size": 16},
    },
    {
        "name": "MyExp3_VeryHighGamma",
        "config_changes": {"learning_rate": 1e-4, "gamma": 0.9999, "batch_size": 32},
    },
]
```

Then run: `conda run -n dqn python train.py`

## What Happens During Training

For each experiment:
1. **Create environment** - Atari Tennis with 84x84 grayscale frames
2. **Train CnnPolicy** - Convolutional neural network (100,000 timesteps)
3. **Save model** - `models/exp_name_cnnpolicy.zip`
4. **Train MlpPolicy** - Fully connected neural network (100,000 timesteps)
5. **Save model** - `models/exp_name_mlppolicy.zip`
6. **Compare** - Show which policy performed better

Final summary shows:
- CnnPolicy wins: X/10
- MlpPolicy wins: Y/10

## Learning Curves & Performance Analysis

### Exp2_LargeBuffer Results (Buffer Size: 100,000)

The data clearly shows **CNN dominance** over MLP:

| Episode Range | CNN Performance | MLP Performance | Difference |
|-------------|----------------|-----------------|-----------|
| 1-55 | -24.0 to -22.0 (exploration) | -24.0 (stuck) | CNN learns faster |
| 56-67 | **-1 to -17** (breakthrough!) | -24.0 (no learning) | CNN: **+23 reward** |
| 68-87 | -10 to -24 (fine-tuning) | -16 to -24 (unstable) | CNN more consistent |

**Key Insight**: Episode 56-67 is the "breakthrough" period where CNN agent suddenly learns effective ball-hitting strategies, while MLP remains stuck at maximum loss. This demonstrates CNN's superior ability to extract meaningful visual features from raw pixels.

### Graph Analysis

From the training visualization:
- **Left panel (Episode Reward)**: Cyan line (CNN) shows dramatic improvement spike; blue line (MLP) flat-lines
- **Right panel (Episode Length)**: Green line (CNN) episode lengths increase dramatically (good — agent stays in game longer), indicating learned skills

![Exp2_LargeBuffer CNN Learning Curve](images/exp2_largebuffer_cnn_curve.png)

## Tips for Better Results

- **Start conservative**: Begin with default parameters, then experiment
- **Change one thing at a time**: Vary learning_rate OR gamma, not both
- **Monitor batch_size**: Smaller batch (16) = more updates, larger batch (128) = more stable
- **Gamma matters**: Higher gamma (0.999) = agent considers far future rewards more
- **Exploration vs exploitation**: Lower epsilon_end (0.01) = greedier agent, higher (0.2) = more random

## Troubleshooting

**Error: "module 'gymnasium.wrappers' has no attribute..."**
- Update gymnasium: `pip install --upgrade gymnasium`

**Error: "No module named 'torch'"**
- Install PyTorch: `pip install torch`

**Error: "No module named 'ale_py'"**
- Install ALE: `pip install ale-py`

**Training is very slow:**
- Make sure you're using the conda environment: `conda run -n dqn python train.py`
- Reduce `total_timesteps` to test quickly (e.g., 10000 instead of 100000)

## Project Structure

```
Deep-Q-Learning-grp11/
├── train.py              # Main training script with 10 experiments
├── README.md             # This file
├── models/               # Saved models (created automatically)
│   ├── exp1_cnnpolicy.zip
│   ├── exp1_mlppolicy.zip
│   └── ...
└── results_*.json        # Training results (created after run)
```

## Team Members

- Group 11 (10 experiments to compare CNN vs MLP policies)

---

**Happy experimenting!**
