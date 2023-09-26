from typing import Tuple
from gym.envs.mujoco.reacher import ReacherEnv as OriginalReacher
from continuous.state_vals import StateVals
from _types import RewardCont, EnvInfoCont, Space

class ReacherEnv(OriginalReacher):
    """
    A customized version of the reacher env allowing you to pass in a custom
    reward function

    Args:
        reward_func: from (self, state, action, next_state) -> reward (float)
    """
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

    def __init__(self, reward_func: RewardCont, discount: float, **kwargs):
        self.init_kwargs = kwargs
        super().__init__(**self.init_kwargs)

        self.reward_func = reward_func
        self.discount = discount
        self.prev_obs = None
        self.state_vals = StateVals(self)

        self.env_info = EnvInfoCont(
            trans_dist=lambda s, a: ReacherEnv.predict_next_state(self, s, a),
            discount=self.discount,
            state_space=ReacherEnv.state_space,
            action_space=ReacherEnv.act_space,
            state_vals=self.state_vals,
        )

    def step(self, *args, **kwargs) -> Tuple:
        """Fixes a non-json-writable element in the info of the base env."""
        obs, _, done, info = super().step(*args, **kwargs)
        reward = self.reward_func(self, self.prev_obs, args[0], obs)
        self.prev_obs = obs

        return obs, reward, done, info
    
    @staticmethod
    def predict_next_state(self, state, action):
        temp_env = OriginalReacher(**self.init_kwargs)

        # Set environment to desired state
        temp_env.set_state(state[:temp_env.model.nq],
                           state[temp_env.model.nq:temp_env.model.nq + temp_env.model.nv])
        
        # Take the desired action
        next_state, _, _, _ = temp_env.step(action)
        
        return next_state
