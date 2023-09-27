from typing import Optional
import numpy as np
import torch
from continuous.env import ReacherEnv
from continuous.rewards.reward_func import RewardFunc
from continuous.rewards.ground_truth_reward import GroundTruthReward
from _types import Space

class PotentialShapedReward(RewardFunc):
    """
        Reward model identical to the ground truth, but with randomly generated
        potential shaping.
    """
    def __init__(self):
        super().__init__()
        self.ground_truth = GroundTruthReward()
        self.potential_weights = np.random.normal(size=len(ReacherEnv.state_space))
        self.potential_bias = np.random.random(1)

    def apply_poten(self, state):
        return np.dot(self.potential_weights, state) + self.potential_bias

    def __call__(self,
                 env: ReacherEnv,
                 state: Optional[torch.Tensor], #TODO fix the types
                 action,
                 next_state) -> float:
        reward = self.ground_truth(env, state, action, next_state)
        if state is not None:
            reward += env.discount * self.apply_poten(next_state)
            reward -= self.apply_poten(state)

        return reward
