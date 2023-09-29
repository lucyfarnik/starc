from typing import Optional
import numpy as np
import torch
from continuous.rewards.reward_func import RewardFunc
from continuous.env import ReacherEnv

class SemanticallyIdenticalReward(RewardFunc):
    """
        Semantically identical to the ground truth, but instead of using a norm
        from the target, it centers a Gaussian over it and then quantizes it.
    """
    std_x = 0.1
    std_y = 0.1

    def __call__(self,
                 env: ReacherEnv,
                 state: Optional[torch.Tensor], #TODO fix the types
                 action,
                 next_state) -> float:
        # evaluate a Gaussian centered on the target
        x, y, _ = env.get_body_com("fingertip")
        x0, y0, _ = env.get_body_com("target")
        # we assume the covariance is the identity matrix
        gauss_val = np.exp(-((x-x0)**2/(2*self.std_x**2) + (y-y0)**2/(2*self.std_y**2)))

        # quantize the Gaussian
        reward = np.round(gauss_val*100)/100

        if hasattr(reward, 'item'): return reward.item()
        return reward
