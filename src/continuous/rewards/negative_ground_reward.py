from typing import Optional
import numpy as np
import torch
from continuous.env import ReacherEnv
from continuous.rewards.reward_func import RewardFunc
from continuous.rewards.ground_truth_reward import GroundTruthReward

class NegativeGroundReward(RewardFunc):
    """
        The rewards are -1 * ground truth reward.
    """
    def __init__(self):
        super().__init__()
        self.ground_truth = GroundTruthReward()

    def __call__(self,
                 env: ReacherEnv,
                 state: Optional[torch.Tensor], #TODO fix the types
                 action,
                 next_state) -> float:
        return -1 * self.ground_truth(env, state, action, next_state)
