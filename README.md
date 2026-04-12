# X-Light

**X-Light** is an official implementation of the research paper:
> *Cross-City Traffic Signal Control Using Transformer on Transformer as Meta Multi-Agent Reinforcement Learner*

This framework trains AI agents to intelligently control traffic signals at multiple intersections simultaneously, and can transfer that knowledge across different city road networks.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Configuration Reference](#configuration-reference)
- [Available Scenarios](#available-scenarios)
- [Project Structure](#project-structure)

---

## Overview

Traffic signal control is a complex real-world problem — poorly timed lights cause congestion, delays, and emissions. X-Light solves this by training a **Multi-Agent Reinforcement Learning (MARL)** system where each traffic light is an AI agent that learns to coordinate with other agents across an entire city network.

**Key capabilities:**
- Controls multiple intersections simultaneously as coordinated agents
- Uses a **Transformer-based** neural network architecture for temporal reasoning
- Supports **co-training** on multiple city scenarios to enable cross-city generalization
- Integrates with [SUMO](https://sumo.dlr.de/), a real-world traffic microsimulator

---

## How It Works

```
Traffic Network (SUMO)
        |
        | observations (queue length, flow, phase, etc.)
        v
+------------------------+
|   Each Traffic Light   |  <-- one RL agent per intersection
|   (Agent)              |
|                        |
|  Transformer Encoder   |  <-- extracts temporal patterns
|        +               |
|  Actor / Critic        |  <-- decides which signal phase to activate
+------------------------+
        |
        | action (choose signal phase 0-7)
        v
Traffic Network updates --> reward computed (reduced wait time / queue)
```

### Algorithm

X-Light uses **MAPPO** (Multi-Agent Proximal Policy Optimization) with three variants:

| Algorithm | Description |
|-----------|-------------|
| `ippo`    | Independent PPO — each agent learns its own value function |
| `mappo`   | Centralized value function shared across all agents |
| `rmappo`  | MAPPO with recurrent (Transformer-based) memory |

Each agent observes local traffic state (vehicle counts, queue length, current phase, etc.) and selects one of 8 signal phases. Agents are rewarded for reducing congestion metrics such as queue length, waiting time, and vehicle delay.

---

## Requirements

- Python 3.6+
- CUDA-capable GPU (recommended)
- SUMO traffic simulator

**Python packages** (see `requirements.txt`):

| Package | Version | Purpose |
|---------|---------|---------|
| `eclipse-sumo` | 1.14.0 | Traffic simulator |
| `torch` | 1.5.0+cu92 | Deep learning |
| `numpy` | 1.19.5 | Numerical computation |
| `wandb` | 0.15.3 | Experiment tracking |
| `gym` | latest | RL environment interface |
| `tensorboardX` | latest | TensorBoard logging |

---

## Installation

### Step 1 — Clone the repository

```bash
git clone <repository-url>
cd x-light
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Set up SUMO environment variable

After installing `eclipse-sumo` via pip, set the `SUMO_HOME` variable to point to the SUMO installation:

```bash
# Linux / macOS
export SUMO_HOME=/path/to/your/python/env/lib/python3.6/site-packages/sumo

# Windows (PowerShell)
$env:SUMO_HOME = "C:\path\to\python\env\Lib\site-packages\sumo"
```

> **Tip:** You can find the correct path by running:
> ```bash
> python -c "import sumo; print(sumo.__path__)"
> ```

### Step 4 — Add the project to PYTHONPATH

```bash
# Linux / macOS
export PYTHONPATH=${PYTHONPATH}:/path/to/x-light

# Windows (PowerShell)
$env:PYTHONPATH += ";C:\path\to\x-light"
```

### Step 5 — Unzip traffic scenarios

```bash
cd onpolicy/envs/sumo_files_marl
unzip scenarios.zip
cd ../../../
```

---

## Quick Start

Run training with default settings (IPPO on grid4x4 scenario):

```bash
python onpolicy/scripts/train/train_sumo.py
```

Training logs will be saved to `onpolicy/scripts/results_sumo/`.

---

## Training

### Single scenario training

To train on a specific scenario, edit the `main()` block at the bottom of `train_sumo.py` and set `cotrain = False`, then set the `index` variable to pick your scenario:

```python
# In train_sumo.py
cotrain = False
index = 0  # 0=grid4x4, 1=fenglin, 2=nanshan, 3=arterial4x4, 4=ingolstadt21, 5=cologne8
```

Then run:

```bash
python onpolicy/scripts/train/train_sumo.py
```

### Co-training (cross-city generalization)

Co-training simultaneously trains the model on multiple city scenarios, enabling it to learn general traffic control strategies that transfer across cities.

Set `cotrain = True` in `train_sumo.py`:

```python
cotrain = True  # Train on all scenarios simultaneously
```

> **Note:** Co-training requires more GPU memory as it runs parallel environments for each scenario.

### Experiment tracking

By default, training logs are sent to **Weights & Biases (W&B)**. To disable and use TensorBoard instead, pass `--use_wandb False` or set `use_wandb=False` in the args list.

---

## Evaluation

To evaluate a pre-trained model, uncomment the evaluation block at the bottom of `train_sumo.py` and set the `model_dir` path:

```python
# In train_sumo.py — evaluation mode
config_env['environment']['is_record'] = True
model_dir = 'onpolicy/scripts/results_sumo/SUMO/{experiment_name}/ippo/seed_0_run1/models'
```

Then run:

```bash
python onpolicy/scripts/train/train_sumo.py
```

The model will be loaded and run in evaluation mode (deterministic policy, no learning updates).

---

## Configuration Reference

All parameters can be passed as command-line arguments or set in the `args` list inside `train_sumo.py`.

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--algorithm_name` | `ippo` | Algorithm: `ippo`, `mappo`, or `rmappo` |
| `--experiment_name` | `check` | Name to identify this experiment run |
| `--seed` | `1` | Random seed for reproducibility |
| `--num_env_steps` | `10,000,000` | Total environment steps to train |
| `--episode_length` | `240` | Steps per episode (≈ 3600 seconds of simulation) |

### Parallelism

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_rollout_threads` | `32` | Number of parallel environments during training |
| `--n_training_threads` | `1` | Number of CPU threads for PyTorch |
| `--n_eval_rollout_threads` | `1` | Number of parallel environments during evaluation |

### Neural Network

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--hidden_size` | `64` | Size of hidden layers in actor/critic networks |
| `--layer_N` | `1` | Number of hidden layers |
| `--use_recurrent_policy` | `True` | Use recurrent (Transformer) policy |
| `--data_chunk_length` | `10` | Sequence length for recurrent training |
| `--share_policy` | `True` | All agents share one policy network |
| `--use_centralized_V` | `True` | Use centralized value function (MAPPO) |

### Optimizer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lr` | `5e-4` | Actor learning rate |
| `--critic_lr` | `5e-4` | Critic learning rate |
| `--use_linear_lr_decay` | `False` | Linearly decay the learning rate over training |

### PPO

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ppo_epoch` | `15` | Number of PPO update epochs per batch |
| `--clip_param` | `0.2` | PPO clipping parameter |
| `--entropy_coef` | `0.01` | Entropy bonus coefficient (encourages exploration) |
| `--gamma` | `0.99` | Discount factor for future rewards |
| `--gae_lambda` | `0.95` | GAE lambda for advantage estimation |
| `--max_grad_norm` | `10.0` | Gradient clipping threshold |

### Saving & Logging

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--save_interval` | `2` | Save model every N episodes |
| `--log_interval` | `5` | Print logs every N episodes |
| `--use_eval` | `False` | Run evaluation alongside training |
| `--eval_interval` | `25` | Evaluate every N episodes |
| `--eval_episodes` | `32` | Number of episodes per evaluation run |
| `--use_wandb` | `True` | Log to W&B (set to False for TensorBoard) |

### SUMO-specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sumocfg_files` | `""` | Path to `.sumocfg` scenario file |
| `--port_start` | `-1` | Starting port for SUMO connections |
| `--cotrain` | `False` | Enable co-training on multiple scenarios |
| `--use_pressure` | `False` | Use pressure-based reward metric |
| `--use_gat` | `False` | Enable Graph Attention Network layers |
| `--model_dir` | `None` | Path to pretrained model for fine-tuning / eval |

---

## Available Scenarios

| Scenario | File | Description |
|----------|------|-------------|
| `grid4x4` | `resco_envs/grid4x4/grid4x4.sumocfg` | 4x4 synthetic grid network |
| `arterial4x4` | `resco_envs/arterial4x4/arterial4x4.sumocfg` | 4x4 arterial road network |
| `ingolstadt21` | `resco_envs/ingolstadt21/ingolstadt21.sumocfg` | Real city: Ingolstadt (21 intersections) |
| `cologne8` | `resco_envs/cologne8/cologne8.sumocfg` | Real city: Cologne (8 intersections) |
| `fenglin` | `sumo_fenglin_base_road/base.sumocfg` | Real city: Fenglin district |
| `nanshan` | `nanshan/osm.sumocfg` | Real city: Nanshan district |
| `large_grid` | `large_grid2/exp_0.sumocfg` | Large-scale synthetic grid |

All scenario files are located under `onpolicy/envs/sumo_files_marl/scenarios/` after unzipping.

---

## Project Structure

```
x-light/
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
└── onpolicy/                           # Main package
    ├── config.py                       # All command-line arguments and defaults
    │
    ├── algorithms/
    │   ├── r_mappo/
    │   │   ├── r_mappo.py              # PPO training logic (loss, updates)
    │   │   └── algorithm/
    │   │       ├── rMAPPOPolicy.py     # Policy wrapper (actor + critic)
    │   │       ├── r_actor_critic.py   # Actor and Critic network definitions
    │   │       ├── sumo_nn.py          # Transformer encoder architecture
    │   │       └── GAT_nn.py           # Graph Attention Network (optional)
    │   └── utils/                      # Reusable network building blocks
    │       ├── mlp.py                  # Multi-layer perceptron
    │       ├── rnn.py                  # Recurrent layers
    │       ├── distributions.py        # Action probability distributions
    │       └── popart.py               # PopArt reward normalization
    │
    ├── envs/
    │   ├── env_wrappers.py             # Parallel environment wrappers
    │   └── sumo_files_marl/
    │       ├── SUMO_env.py             # Main SUMO environment wrapper
    │       ├── config.py               # Environment-specific configuration
    │       ├── scenarios.zip           # Pre-built traffic scenarios (unzip first)
    │       └── env/
    │           ├── sim_env.py          # Core simulation engine (TSCSimulator)
    │           └── intersection.py     # Single intersection logic and observations
    │
    ├── runner/
    │   └── shared/
    │       ├── base_runner.py          # Base training loop (setup, logging, checkpointing)
    │       └── sumo_runner.py          # SUMO-specific training loop
    │
    ├── scripts/
    │   └── train/
    │       └── train_sumo.py           # Entry point — run this to start training
    │
    └── utils/
        ├── shared_buffer.py            # Experience replay buffer with GAE returns
        ├── valuenorm.py                # Running mean/std value normalization
        └── util.py                     # Miscellaneous helpers
```

### Key Files to Know

| File | Role |
|------|------|
| `train_sumo.py` | **Start here** — configures and launches training |
| `sumo_runner.py` | Orchestrates the training loop episode-by-episode |
| `r_mappo.py` | Computes PPO loss and updates neural network weights |
| `sumo_nn.py` | Defines the Transformer encoder that processes observations |
| `sim_env.py` | Interfaces directly with the SUMO simulator |
| `intersection.py` | Models a single traffic light's state and observation |
| `shared_buffer.py` | Stores and samples experience for training |

---

## Reward Metrics

Each agent receives a reward at every timestep based on traffic conditions at its intersection. The following metrics are supported (configured via `state_key` in environment config):

| Metric | Description |
|--------|-------------|
| `queue_length` | Number of vehicles queued at red lights |
| `wait_time` | Total time vehicles spend waiting |
| `car_num` | Number of vehicles in nearby lanes |
| `occupancy` | Lane occupancy percentage |
| `flow` | Vehicle throughput |
| `stop_car_num` | Number of stopped vehicles |
| `pressure` | Difference in vehicle count between incoming and outgoing lanes |
| `current_phase` | Current active signal phase |

---

## Citation

If you use X-Light in your research, please cite the original paper (refer to the paper for full citation details).

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
