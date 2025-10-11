# 2025 NeurIPS - MyoChallenge

Welcome to the [**2025 NeurIPS - MyoChallenge: Towards Human Athletic Intelligence**](https://sites.google.com/view/myosuite/myochallenge/myochallenge-2025).

This challenge consists of developing controllers for a physiologically realistic musculoskeletal model to achieve upper and lower limb athletic tasks:

- A) **PingPong task** -- Successfully tally a incoming pingpong ball (`myoChallengeTableTennisP1-v0`).

## Build
### Base image
```bash
docker build \
  -t ghcr.io/rtae/myochallenge/myochallenge-base:latest \
  -f Dockerfile.base .
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

### Run with Docker 
```bash
docker run -it --rm \
  --gpus all \
  -v $PWD:/app \
  ghcr.io/rtae/myochallenge/myochallenge-base:latest \
  bash
```

## Train
1. Run with Docker
```bash
docker run -d -name myochallenge_train \
  --gpus all \
  -v ./logs:/app/logs \
  ghcr.io/rtae/myochallenge/myochallenge-train:latest
```