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
        
        touching_info = defaultdict(list)

        for info in infos:
            for k, v in info.items():
                if isinstance(v, (int, float, np.number)):
                    acc[k].append(float(v))
                

                # touching_info
                if k == "obs_dict":
                    touching_array = v.get("touching_info", None)
                    if touching_array is not None:
                        touching_info["touching_info_paddle"].append(float(touching_array[0]))
                        touching_info["touching_info_own"].append(float(touching_array[1]))
                        touching_info["touching_info_opponent"].append(float(touching_array[2]))
                        touching_info["touching_info_ground"].append(float(touching_array[3]))
                        touching_info["touching_info_net"].append(float(touching_array[4]))
                        touching_info["touching_info_env"].append(float(touching_array[5]))

        for k, vals in acc.items():
            self.logger.record(f"{self.prefix}/{k}_mean", np.mean(vals))
            
        for k, vals in touching_info.items():
            self.logger.record(f"{self.prefix}/{k}_mean", np.mean(vals))

        return True
