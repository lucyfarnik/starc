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
    def __init__(self, env: ReacherEnv, state_space):
        super().__init__(env)
        self.ground_truth = GroundTruthReward(env)

    def __call__(self,
                 state: Optional[torch.Tensor], #TODO fix the types
                 action,
                 next_state) -> float:
        return -1 * self.ground_truth(state, action, next_state)
