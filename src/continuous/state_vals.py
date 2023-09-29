import torch
from continuous.env import ReacherEnv
from continuous.sarsa import train_sarsa_model
from continuous.sarsa import device as sarsa_device

class StateVals:
  """
  A class for computing the state values of a PPO agent
  """
  def __init__(self,
               env: ReacherEnv,
               reward_class_name: str,
               n_episodes_sarsa: int = 10000):
    self.model = train_sarsa_model(env, reward_class_name, n_episodes_sarsa)

    for param in self.model.parameters():
      param.requires_grad = False

  def __call__(self, state):
    obs_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(sarsa_device)

    return self.model(obs_tensor).item()
