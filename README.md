# MyoChallenge2025 Table Tennis Rally  - Synergy-Driven Hierarchical Policies

In this study, we investigate the NeurIPS 2025 MyoChallenge Table Tennis Rallytask,  where  an  agent  must  control  a  high-dimensional  musculoskeletal  modelto  return  a  ball  to  the  opponent’s  side  of  the  table.   The  key  difficulty  is  ex-ploration  in  an  overactuated  control  space;  naive  deep  reinforcement  learning(DeepRL)  methods  such  as  SAC  or  PPO  often  fail  to  discover  coherent,  task-relevant  muscle  activation  patterns.   Our  work  combines  exploration  methodsfor overactuated systems with a hierarchical controller.  The low-level policy istrained to move the paddle toward spatial sub-goals in 3D using structured ex-ploration  that  encourages  muscle  synergies,  while  the  high-level  policy  selectsthose  sub-goals  over  time.   Our  objective  is  to  achieve  a  relatively  strong  per-formance  and  empirically  evaluate  whether  this  combined  approach  improvesexploration  and  control  quality  in  the  MyoChallenge  table  tennis  environmentcompared  to  deep  reinforcement  learning  baselines.  

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