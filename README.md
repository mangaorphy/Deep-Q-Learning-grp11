# Deep Q-Learning (DQN) - Atari Tennis Experiments

## Overview

This project trains Deep Q-Network (DQN) agents to play Atari Tennis using both **CnnPolicy** and **MlpPolicy**, comparing their performance across different hyperparameter configurations.

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

**Happy experimenting!** 🚀
