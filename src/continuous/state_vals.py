import torch
from stable_baselines3 import PPO
from stable_baselines3.ppo import PPO, MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from continuous.env import ReacherEnv


class StateVals:
  """
  A class for computing the state values of a PPO agent
  """
  def __init__(self, env: ReacherEnv):
    self.env = env
    model = PPO(MlpPolicy,
                DummyVecEnv([lambda: env]),
                verbose=1,
                gamma=env.discount)

    # Train the agent
    model.learn(total_timesteps=2000)
    
    self.device = model.device
    self.policy = model.policy

    for _, param in self.policy.named_parameters():
      param.requires_grad = False

  def __call__(self, state):
    obs_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

    return self.policy.predict_values(obs_tensor).item()
