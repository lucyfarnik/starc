from typing import Optional
import numpy as np
import torch
from continuous.rewards.reward_func import RewardFunc
from continuous.env import ReacherEnv

class GroundTruthReward(RewardFunc):
    """
        This is the original reward function from the ReacherEnv class.
    """
    def __call__(self,
                 env: ReacherEnv,
                 state: Optional[torch.Tensor], #TODO fix the types
                 action,
                 next_state) -> float:
        vec = env.get_body_com("fingertip") - env.get_body_com("target")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(action).sum()
        reward = reward_dist + reward_ctrl

        if hasattr(reward, 'item'): return reward.item()
        return reward

# class GroundTruthReward(RewardModel):
#     """Reward model for custom Reacher environment.

#     Args:
#         obs_space: The observation space used in the environment.
#         act_space: The action space used in the environment.
#         reward_dist_factor: Weight on the distance from goal reward term.
#         reward_ctrl_factor: Weight on the control reward term.
#         reward_goal_factor: Weight on reaching the goal.
#         shaping_factor: The value to scale the shaping.
#         shaping_discount: The discount factor used in potential shaping.
#     """
#     # At this threshold around 2% of initial states are next to the goal.
#     GOAL_REACHED_THRESHOLD = 0.05

#     def __init__(
#             self,
#             obs_space: gym.spaces.Space,
#             act_space: gym.spaces.Space,
#             reward_dist_factor: float,
#             reward_ctrl_factor: float,
#             reward_goal_factor: float,
#             shaping_factor: float,
#             shaping_discount: float,
#     ):
#         self.obs_space = obs_space
#         self.act_space = act_space
#         self.reward_dist_factor = reward_dist_factor
#         self.reward_ctrl_factor = reward_ctrl_factor
#         self.reward_goal_factor = reward_goal_factor
#         self.shaping_factor = shaping_factor
#         self.shaping_discount = shaping_discount

#     @property
#     def observation_space(self) -> gym.spaces.Space:
#         return self.obs_space

#     @property
#     def action_space(self) -> gym.spaces.Space:
#         return self.act_space

#     def reward(
#             self,
#             states: torch.Tensor,
#             actions: torch.Tensor,
#             next_states: Optional[torch.Tensor],
#             terminals: Optional[torch.Tensor],
#     ) -> torch.Tensor:
#         """Computes the reward for the environment.

#         See base class for documentation on args and return value.
#         """
#         del terminals
#         states_dists = states[:, -3:].norm(dim=-1, keepdim=True)
#         dist_rewards = -states_dists
#         ctrl_rewards = -actions.square().sum(dim=1, keepdim=True).to(states.dtype)
#         goal_rewards = states_dists < self.GOAL_REACHED_THRESHOLD

#         next_states_dists = next_states[:, -3:].norm(dim=-1, keepdim=True)
#         shaping_rewards = (self.shaping_discount * next_states_dists - states_dists)

#         rewards = self.reward_dist_factor * dist_rewards \
#             + self.reward_ctrl_factor * ctrl_rewards \
#             + self.reward_goal_factor * goal_rewards \
#             + self.shaping_factor * shaping_rewards

#         return rewards
