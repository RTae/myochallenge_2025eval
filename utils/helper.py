import os

def next_exp_dir(base="./logs") -> str:
    os.makedirs(base, exist_ok=True)
    exps = [int(d[3:]) for d in os.listdir(base) if d.startswith("exp") and d[3:].isdigit()]
    exp_dir = os.path.join(base, f"exp{max(exps)+1 if exps else 1}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir
