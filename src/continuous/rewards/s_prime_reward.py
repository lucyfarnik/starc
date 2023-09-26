from typing import Optional
import numpy as np
import torch
from continuous.env import ReacherEnv
from continuous.rewards.reward_func import RewardFunc
from continuous.rewards.ground_truth_reward import GroundTruthReward
from continuous.env import predict_next_state

class SPrimeReward(RewardFunc):
    """
        Returns the ground truth if s' follows from s,a; otherwise returns a
        very large reward (because teleporting is cool lol).

        Note that this is semantically equivalent to the ground truth
        since the "teleporting" transitions are impossible.
    """
    def __init__(self, env: ReacherEnv):
        super().__init__(env)
        self.ground_truth = GroundTruthReward(env)
    
    def __call__(self,
                 state: Optional[torch.Tensor], #TODO fix the types
                 action,
                 next_state) -> float:
        if next_state == predict_next_state(self.env, state, action):
            return self.ground_truth(state, action, next_state)
        
        return 1e6
