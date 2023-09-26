from typing import Tuple
from gym.envs.mujoco.reacher import ReacherEnv as OriginalReacher
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.ppo import PPO, MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from _types import RewardCont

class ReacherEnv(OriginalReacher):
    """
    A customized version of the reacher env allowing you to pass in a custom
    reward function

    Args:
        reward_func: from (self, state, action, next_state) -> reward (float)
    """
    def __init__(self, reward_func: RewardCont, discount: float, **kwargs):
        super().__init__(**kwargs)
        self.reward_func = reward_func
        self.discount = discount
        self.prev_obs = None

    def step(self, *args, **kwargs) -> Tuple:
        """Fixes a non-json-writable element in the info of the base env."""
        obs, _, done, info = super().step(*args, **kwargs)
        reward = self.reward_func(self, self.prev_obs, args[0], obs)
        self.prev_obs = obs

        return obs, reward, done, info

def get_vec_env() -> DummyVecEnv:
    """
        Wrap your custom environment. VecEnvs are typically used for better performance.
    """
    return DummyVecEnv([lambda: ReacherEnv(obs_mode='original')])

def train_agent(env_vec: DummyVecEnv, discount: float) -> PPO:
    # Instantiate the agent
    model = PPO(MlpPolicy, env_vec, verbose=1, gamma=discount)
    # Train the agent
    model.learn(total_timesteps=2000)

    return model

def predict_next_state(env: ReacherEnv, state, action):
    #! RIGHT NOW THIS HAS THE SIDE EFFECT OF CHANGING THE STATE — need to back it up and restore it

    # Set environment to desired state
    env.set_state(state[:env.model.nq], state[env.model.nq:env.model.nq + env.model.nv])
    
    # Take the desired action
    next_state, _, _, _ = env.step(action)
    
    return next_state

def state_val(model: PPO, state):
    obs_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(model.device)

    return model.policy.predict_values(obs_tensor).item()
