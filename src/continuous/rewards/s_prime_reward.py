from typing import Optional
import numpy as np
import torch
from continuous.env import ReacherEnv
from continuous.rewards.reward_func import RewardFunc
from continuous.rewards.ground_truth_reward import GroundTruthReward
from continuous.rewards.random_reward import RandomReward
from continuous.env import ReacherEnv

class SPrimeReward(RewardFunc):
    """
        Returns the ground truth if s' follows from s,a; otherwise returns a
        very large reward (because teleporting is cool lol).

        Note that this is semantically equivalent to the ground truth
        since the "teleporting" transitions are impossible.
    """
    def __init__(self):
        super().__init__()
        self.ground_truth = GroundTruthReward()
        self.random_reward = RandomReward()
    
    def __call__(self,
                 env: ReacherEnv,
                 state: Optional[torch.Tensor], #TODO fix the types
                 action,
                 next_state) -> float:
        if state is None or np.allclose(next_state,
                                        ReacherEnv.predict_next_state(state, action),
                                        rtol=1e-3, atol=1e-3):
            return self.ground_truth(env, state, action, next_state)
        
        return self.random_reward(env, state, action, next_state)
