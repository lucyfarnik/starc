from typing import Optional
import numpy as np
import torch
from continuous.env import ReacherEnv
from continuous.rewards.reward_func import RewardFunc
from continuous.rewards.ground_truth_reward import GroundTruthReward
from utils import sample_space

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
        self.target_position = env.get_body_com("target")[:2]
        self.second_peak = sample_space(ReacherEnv.state_space[4:6])
        while np.linalg.norm(self.second_peak - self.target_position) < 1:
            self.second_peak = sample_space(ReacherEnv.state_space[4:6])
         
    def __call__(self,
                 env: ReacherEnv,
                 state: Optional[torch.Tensor], #TODO fix the types
                 action,
                 next_state) -> float:

        # if there currently isn't a second peak, or if the target has moved,
        # create a new second peak
        if self.second_peak is None or not np.allclose(env.get_body_com("target")[:2],
                                                       self.target_position,
                                                       rtol=1e-3):
            self.create_second_peak(env)
        

        reward = self.ground_truth(env, state, action, next_state)
        reward += -0.2*np.linalg.norm(self.second_peak - env.get_body_com("fingertip")[:2])

        return reward
