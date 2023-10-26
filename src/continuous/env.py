import numpy as np
from typing import Tuple
from functools import partial
from gym.envs.mujoco.reacher import ReacherEnv as OriginalReacher
from _types import EnvInfoCont, Space

class ReacherEnv(OriginalReacher):
    """
    A customized version of the reacher env allowing you to pass in a custom
    reward function

    Args:
        reward_func: from (self, state, action, next_state) -> reward (float)
    """
    original_env_instance = OriginalReacher()

    state_space: Space = [
        (-1, 1), # cosine of the angle of the first arm
        (-1, 1), # cosine of the angle of the second arm
        (-1, 1), # sine of the angle of the first arm
        (-1, 1), # sine of the angle of the second arm
        (-0.5, 0.5), # x-coordinate of the target
        (-0.5, 0.5), # y-coordinate of the target
        (-10.5, 10.5), # angular velocity of the first arm
        (-10.5, 10.5), # angular velocity of the second arm
        (-1, 1), # x-value of position_fingertip - position_target
        (-1, 1), # y-value of position_fingertip - position_target
        (0, 0), # z-value of position_fingertip - position_target (0 since reacher is 2d and z is same for both)
    ]
    act_space: Space = [
        (-1, 1), # Torque applied at the first hinge (connecting the link to the point of fixture)
        (-1, 1), # Torque applied at the second hinge (connecting the two links)
    ]

    def __init__(self, reward_func, discount: float, n_episodes_sarsa: int = 10000, **kwargs):
        self.reward_func = reward_func
        self.reward_func_curried = partial(reward_func, self)
        self.discount = discount
        self.prev_obs = None
        super().__init__(**kwargs)

        # TODO (low-priority): refactor to avoid circular imports — don't have state_vals in env
        from continuous.state_vals import StateVals
        self.state_vals = StateVals(self,
                                    self.reward_func.__class__.__name__,
                                    n_episodes_sarsa)

        # utility class allowing us to pass around the information about the env easily
        self.env_info = EnvInfoCont(
            trans_dist=ReacherEnv.predict_next_state,
            trans_dist_deterministic=True,
            discount=self.discount,
            state_space=ReacherEnv.state_space,
            action_space=ReacherEnv.act_space,
            state_vals=self.state_vals,
            state_vals_deterministic=True,
        )

    def step(self, *args, **kwargs) -> Tuple:
        obs, _, done, info = super().step(*args, **kwargs)
        reward = self.reward_func(self, self.prev_obs, args[0], obs)
        self.prev_obs = obs

        return obs, reward, done, info
    
    @staticmethod
    def predict_next_state(state, action):
        # if the arrays aren't already numpy arrays, convert them
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        
        env = ReacherEnv.original_env_instance

        # Set environment to desired state
        env.set_state(state[:env.model.nq],
                      state[env.model.nq : env.model.nq + env.model.nv])
        
        # Take the desired action
        next_state, _, _, _ = env.step(action)
        
        return next_state
