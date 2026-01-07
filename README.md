# 2025 NeurIPS - MyoChallenge

Welcome to the [**2025 NeurIPS - MyoChallenge: Towards Human Athletic Intelligence**](https://sites.google.com/view/myosuite/myochallenge/myochallenge-2025).

This challenge consists of developing controllers for a physiologically realistic musculoskeletal model to achieve upper and lower limb athletic tasks:

- A) **PingPong task** -- Successfully tally a incoming pingpong ball (`myoChallengeTableTennisP1-v0`).

## Demo
### Baseline PPO


## Build
### Base image (Optional)
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


## Deverlopment
### Run with local python
1. Create uv environment with package dependencies
```bash
uv install
uv sync
```

2. Run training script
```bash
python train.py
```

## Train with Docker

### Prerequisites
1. Make sure you already pull these image below into your local.
```bash
docker pull tensorflow/tensorflow:2.15.0
```
2. Install docker gpu https://docs.docker.com/engine/containers/resource_constraints/#gpu

### Steps
1. Edit *Dockerfile.train* to train specific task
Change the line below to your desired task
```Dockerfile
CMD ["python", "train.py"]
```
to
```Dockerfile
CMD ["python", "train_ppo_hrl_lattice.py"]
```
2. Build Docker image
```bash
make build
```
3. Run training container
```bash
make train
```
4. Stop training container (optional)
You don't need to do it everytime you can just run *make train* again after build a image from *make build*
```bash
make stop
```

## Evaluation