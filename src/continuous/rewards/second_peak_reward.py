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
    def __init__(self):
        super().__init__()
        self.ground_truth = GroundTruthReward()
        self.second_peak = None
    
    def create_second_peak(self, env):
        # pick a random point in the space that's not too close to the target
        target_position = env.get_body_com("target")
        self.second_peak = np.random.uniform(*ReacherEnv.state_space, size=2)
        while np.linalg.norm(self.second_peak - target_position) < 1:
            self.second_peak = np.random.uniform(*ReacherEnv.state_space, size=2)
         
    def __call__(self,
                 env: ReacherEnv,
                 state: Optional[torch.Tensor], #TODO fix the types
                 action,
                 next_state) -> float:

        # if there currently isn't a second peak, or if the target has moved,
        # create a new second peak
        if self.second_peak is None or \
              not np.allclose(env.get_body_com("target"), self.target_position):
                self.create_second_peak(env)
        

        reward = self.ground_truth(state, action, next_state)
        reward += -0.2*np.linalg.norm(self.second_peak - env.get_body_com("fingertip"))

        return reward
