from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict
import numpy as np

class InfoLoggerCallback(BaseCallback):
    def __init__(self, prefix="train/info", verbose=0):
        super().__init__(verbose)
        self.prefix = prefix

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        acc = defaultdict(list)

        for info in infos:
            for k, v in info.items():
                if isinstance(v, (int, float, np.number)):
                    acc[k].append(float(v))

        for k, vals in acc.items():
            self.logger.record(f"{self.prefix}/{k}_mean", np.mean(vals))

        return True
