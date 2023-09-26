import torch
from stable_baselines3 import PPO
from stable_baselines3.ppo import PPO, MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from continuous.env import ReacherEnv

class StateVals:
  """
  A class for computing the state values of a PPO agent
  """
  def __init__(self, env: ReacherEnv):
    self.env = env
    self.model = PPO(MlpPolicy,
                     DummyVecEnv([lambda: env]),
                     verbose=1,
                     gamma=env.discount)
    # Train the agent
    self.model.learn(total_timesteps=2000)

  def __call__(self, state):
    obs_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.model.device)

    return self.model.policy.predict_values(obs_tensor).item()
