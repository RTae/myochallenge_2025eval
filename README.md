# MyoChallenge 2025 Table Tennis Rally — Synergy-Driven Hierarchical Policies

This project studies the **NeurIPS 2025 MyoChallenge Table Tennis Rally** task, where an agent controls a high-dimensional musculoskeletal model to return a ball to the opponent’s side. The main challenge is **exploration in an overactuated action space**: standard Deep RL methods such as **SAC** or **PPO** often struggle to discover coherent, task-relevant muscle activation patterns.

We combine **structured exploration for overactuated systems** with a **hierarchical controller**. The **low-level policy** learns to move the paddle toward 3D spatial sub-goals using exploration that encourages muscle synergies, while the **high-level policy** selects those sub-goals over time. Our goal is to achieve strong performance and empirically evaluate whether this combined approach improves exploration and control quality compared to Deep RL baselines.


## Demo

### Baseline PPO
<div style="display:flex; gap:12px; justify-content:center; align-items:flex-start;">
  <img src="https://raw.githubusercontent.com/RTae/myochallenge_2025eval/main/assert/ppo_1.gif" width="360" />
  <img src="https://raw.githubusercontent.com/RTae/myochallenge_2025eval/main/assert/ppo_2.gif" width="360" />
</div>

### Our Method (Hierarchical Policy with Synergy-Driven Exploration)
<div style="display:flex; gap:12px; justify-content:center; align-items:flex-start;">
  <img src="https://raw.githubusercontent.com/RTae/myochallenge_2025eval/main/assert/our_1.gif" width="360" />
  <img src="https://raw.githubusercontent.com/RTae/myochallenge_2025eval/main/assert/our_1.gif" width="360" />
</div>

## Build

### Base image (optional)
```bash
docker build \
  -t ghcr.io/rtae/myochallenge/myochallenge-base:latest \
  -f Dockerfile.base .
```

### Train image
```bash
docker build \
  -t ghcr.io/rtae/myochallenge/myochallenge-train:latest \
  -f Dockerfile.train .
```

---

## Development environment
### Run locally (Python)
1.	Create a uv environment and install dependencies.
```bash
uv install
uv sync
```

2. Run training script.
```bash
python train.py
```
Plan: I’ll rewrite your README section with cleaner wording + consistent headings, keep your side-by-side demo, and fix small typos (task name, “Development”, etc.) without changing meaning.

## Train a model
### Prerequisites
1.	Pull the required base image:
```bash
docker pull tensorflow/tensorflow:2.15.0
```

2.	Install Docker GPU support:
- https://docs.docker.com/engine/containers/resource_constraints/#gpu

### Steps
1.	Choose the training entrypoint in Dockerfile.train.
Change:
```Dockerfile
CMD ["python", "train.py"]
```
To one of the following:
```Dpockerfile
CMD ["python", "train_ppo_hrl_lattice.py"]
```
Available scripts:
- train_ppo_hrl_lattice.py
- train_ppo_hrl.py
- train_ppo_lattice.py
- train_ppo.py
- train_sac_hrl_lattice.py
- train_sac_hrl.py
- train_sac_lattice.py
- train_sac.py

Also, you can adjust a training configuration in `configs.py` or directly in each training script.

2.	Build the Docker image:
```bash
make build
```
3.	Start training:
After running a script you can check a log file in `logs/` directory and also TensorBoard.
```bash
make train
```
4.	Stop training (optional):
```bash
make stop
```

## Evaluation
```bash
python scripts/eval_hrl_recurrentppo.py \
  --logs ./logs \
  --glob "ppo_hrl_lattice*/" \
  --trials 200 \
  --use-best \
  --eval-worker
```
Example output:
```
Found 1 experiment folders. Beginning evaluation...

--------------------------------------------------------------------------------
Evaluating: 100%|████████████████████████████| 200/200 [00:41<00:00,  4.90it/s]
Finished ppo_hrl_lattice_seed0: HighLevelR=18.42, HighLevelS=7.5%

==========================================================================================
FINAL AGGREGATED REPORT (High Level Policy)
==========================================================================================
           Experiment  High Level Policy Mean Reward  High Level Policy Std Reward  High Level Policy Success Rate (%)  High Level Policy Mean Effort  High Level Policy Std Effort
ppo_hrl_lattice_seed0                        18.42                      5.12                              6.20                         0.18                     0.0410
------------------------------------------------------------------------------------------
High Level Policy Reward:       18.42 ± 5.12
High Level Policy Success Rate: 6.20% ± 0.18
==========================================================================================
```