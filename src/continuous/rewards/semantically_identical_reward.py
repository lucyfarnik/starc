from typing import Optional
import numpy as np
import torch
from continuous.rewards.reward_func import RewardFunc

class SemanticallyIdenticalReward(RewardFunc):
    """
        Semantically identical to the ground truth, but instead of using a norm
        from the target, it centers a Gaussian over it and then quantizes it.
    """
    def __call__(self,
                 state: Optional[torch.Tensor], #TODO fix the types
                 action,
                 next_state) -> float:
        # evaluate a Gaussian centered on the target
        x, y = self.env.get_body_com("fingertip")
        x0, y0 = self.env.get_body_com("target")
        # we assume the covariance is the identity matrix
        gauss_val = np.exp(-((x-x0)**2/2 + (y-y0)**2/2))

        # quantize the Gaussian
        return np.round(gauss_val*10)/10
