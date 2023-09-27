from typing import Optional
import numpy as np
import torch
from continuous.env import ReacherEnv
from continuous.rewards.reward_func import RewardFunc
from _types import Space

class RandomReward(RewardFunc):
    """
        Returns random (but deterministic) rewards
    """
    def __init__(self):
        self.s_weights = np.random.normal(size=len(ReacherEnv.state_space))
        self.a_weights = np.random.normal(size=len(ReacherEnv.act_space))
        self.s_prime_weights = np.random.normal(size=len(ReacherEnv.state_space))
        self.bias = np.random.random(1)

    def __call__(self,
                 env: ReacherEnv,
                 state: Optional[torch.Tensor], #TODO fix the types
                 action,
                 next_state) -> float:
        if state is None:
            return 0
        
        reward = np.dot(self.s_weights, state) + np.dot(self.a_weights, action) \
            + np.dot(self.s_prime_weights, next_state) + self.bias

        return reward
