# FBiT: A Bidirectional Topology-Factored Reinforcement Learning Framework with Coordination Demand Forecasting for Supply Chain Control

## Repository structure

```
algos/                      # Algorithms
  FBiT.py  

supply_chain_env/           # Supply chain environments
  __init__.py               # Registers env id: "SupplyChainG-v0"
  envs/
    supply_chain_g_v0.py   
    config/
      v2_config.json        # Environment configuration

results/                    # Experiment outputs
  SupplyChainG-v0/
  SupplyChainGReal-v0/

utils/                      # Utilities and plotting
  util.py  
  plotting_*.py             # Visualization scripts
```

## Installation

Ensure you have Python 3.8+ (PyTorch and PyG versions in `requirements.txt` may constrain this).

```powershell
pip install -r requirements.txt
```

Optional: a CUDA-capable GPU is recommended for faster training; CPU is supported.

## Quick start

The main entry point is `algos/FBiT.py`, which defines the `SAC_FBiT` agent and can be run directly.

### Option A — Run as a script

```powershell
python .\algos\FBiT.py
```

This will train on the registered environment id `SupplyChainG-v0` and save outputs under `results/`.

### Option B — Use as a library

```python
from algos.FBiT import SAC_FBiT
from utils.util import prepare_training_env

env_id = "SupplyChainG-v0"
env_mode = 1

# Create vectorized training environment
env = prepare_training_env(env_id, env_mode=env_mode, seed=7548)

agent = SAC_FBiT(epi_num=5000)
agent.learn(env_mode=env_mode, env=env)

# Optional evaluation
eval_env = prepare_training_env(env_id, env_mode=env_mode, seed=200)
agent.test(env_mode=env_mode, env=eval_env, n_eval_episodes=100)
```

Notes:

- The environment is registered in `supply_chain_env/__init__.py` with id `SupplyChainG-v0`.
- Checkpointing and TensorBoard logs are written under `results/.../learn*/`.

## Evaluation on gym-invmgmt

Code for integrating and evaluating FBiT on the external `gym-invmgmt`
benchmark is available in [`gym_invmgmt_benchmark/`](gym_invmgmt_benchmark/).

## Troubleshooting

- If GPU is available but not used, verify your PyTorch CUDA installation matches your driver/CUDA runtime.
- If PyTorch Geometric (PyG) raises version errors, ensure the versions in `requirements.txt` are installed and compatible with your PyTorch build.

## License

This project is distributed under the terms specified in the `LICENSE` file.
