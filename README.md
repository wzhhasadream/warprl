# WarpSAC: Towards the Pinnacle of Scalable Off-Policy Reinforcement Learning by Rethinking Exploration and Exploitation

This repository contains the official implementation of WarpSAC, together with the training scripts and environment integrations used in our experiments. WarpSAC extends FlashSAC with regime-aware replay and stabilization choices, improving sample efficiency and training wall-clock efficiency in the evaluated settings.

## Demonstrations

[Open the WarpSAC demo video](https://github.com/wzhhasadream/warprl/raw/refs/heads/main/assets/warpsac.mp4)

<video controls muted loop playsinline width="720" src="https://github.com/wzhhasadream/warprl/raw/refs/heads/main/assets/warpsac.mp4">
  Your browser does not support embedded video. Use the link above.
</video>


![Unitree G1 sim-to-real learning curves](./assets/unitree_g1_flat_learning_curves.png)

The framework supports JAX and PyTorch backends. The reported WarpSAC results use the JAX backend. The PPO implementation uses the PyTorch backend with mixed-precision execution and `torch.compile`; this is an optimized implementation built on top of the RSL-RL training pipeline, whose standard configuration does not provide these two optimizations.

## Installation

Create the base environment:

```bash
conda env create -f environment.yml
conda activate warp_rl
```

The IsaacSim and IsaacLab stack is optional. Reserve at least 50 GB of free disk space for these packages:

```bash
conda activate warp_rl
python -m pip install --extra-index-url https://pypi.nvidia.com \
  "isaacsim[all,extscache]==5.1.0" \
  "isaaclab[isaacsim]==2.3.0"
```

Verify the installation with:

```bash
import jax
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda, "available:", torch.cuda.is_available())
print("JAX:", jax.__version__, "devices:", jax.devices())
```

## Training

All commands are run from the repository root. Use `--seed` to select an experiment seed.

### WarpSAC on CPU-scale MuJoCo

```bash
python train_warpsac.py \
  --env-id Hopper-v4 \
  --env-type mujoco \
  --backend jax \
  --seed 1
```

### WarpSAC on GPU-parallel IsaacLab

```bash
CUDA_VISIBLE_DEVICES=0 python train_warpsac.py \
  --env-id Isaac-Velocity-Flat-G1-v0 \
  --env-type isaaclab \
  --backend jax \
  --seed 1
```

### Unitree G1 sim-to-real

The `Unitree-G1-Flat` profile uses the MJLab environment and the sim-to-real configuration defined by the training script.

```bash
CUDA_VISIBLE_DEVICES=0 python train_warpsac.py \
  --env-id Unitree-G1-Flat \
  --env-type mjlab \
  --backend jax \
  --seed 1
```

### PPO on Unitree G1 sim-to-real

```bash
CUDA_VISIBLE_DEVICES=0 python train_ppo.py \
  --env-id Unitree-G1-Flat \
  --env-type mjlab \
  -- backend torch \
  --seed 1
```

### Important WarpSAC options

The main controls that distinguish the WarpSAC sampling and normalization regimes are:

```text
--decay-step 0
--actor-normalize-parameters
--critic-normalize-parameters
```

`decay_step=0` disables the decay-based sampling bias and recovers uniform replay. Parameter normalization is enabled by default; for large GPU-parallel simulations, disabling actor and critic parameter normalization is recommended unless the experiment requires otherwise.
