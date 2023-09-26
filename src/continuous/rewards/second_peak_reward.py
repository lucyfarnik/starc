from typing import Optional, Tuple
import numpy as np
import torch
from continuous.env import ReacherEnv
from continuous.rewards.reward_func import RewardFunc
from continuous.rewards.ground_truth_reward import GroundTruthReward

class SecondPeakReward(RewardFunc):
    """
        Add a second, smaller peak somewhere else in the Cartesian space.

        Since the peak is smaller, the optimal policy shouldn't change much,
        so we'd expect the distance to the ground truth to be small.
    """
    def __init__(self, env: ReacherEnv, space_bounds: Tuple[float, float]):
        super().__init__(env)
        self.ground_truth = GroundTruthReward(env)

        # pick a random point in the space that's not too close to the target
        target_position = self.env.get_body_com("target")
        self.second_peak = np.random.uniform(*space_bounds, size=2)
        while np.linalg.norm(self.second_peak - target_position) < 1:
            self.second_peak = np.random.uniform(*space_bounds, size=2)

    def __call__(self,
                 state: Optional[torch.Tensor], #TODO fix the types
                 action,
                 next_state) -> float:
        reward = self.ground_truth(state, action, next_state)
        reward += -0.2*np.linalg.norm(self.second_peak - self.env.get_body_com("fingertip"))

        return reward
