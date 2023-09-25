# Same starting point as mujoco_reacher.ipynb, we're on the same Linux instance
# rn so just copied it so we can both edit
# %%
import gym

# %%
import abc
from typing import Optional

import gym
import torch


class RewardModel(abc.ABC):
    @abc.abstractmethod
    def reward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: Optional[torch.Tensor],
            terminals: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Computes the reward for the associated transitions.

        We assume that all reward models operate on `torch.Tensor`s.

        Args:
            states: The states.
            actions: The actions.
            next_states: The next states. Some reward models don't use these so they're optional.
            terminals: Indicators for whether the transition ended an episode.
                Some reward models don't use these so they're optional.

        Returns:
            Tensor of scalar reward values.
        """
    @property
    @abc.abstractmethod
    def observation_space(self) -> gym.spaces.Space:
        """Returns the observation space of this reward model."""
    @property
    @abc.abstractmethod
    def action_space(self) -> gym.spaces.Space:
        """Returns the action space of this reward model."""


# %%
from typing import Optional, Tuple

import gym
from gym.envs.mujoco.reacher import ReacherEnv
import numpy as np
import torch
import mujoco_py


class CustomReacherEnv(ReacherEnv):
    """A customized version of the reacher env.

    Customization includes frame skip, changing the obs to allow for simulation from it,
    making the info dict json serializable, setting a finite horizon independent of the
    gym wrapper for doing so, and other changes.

    Args:
        frame_skip: Number of frames to skip between timesteps.
        max_timesteps: The maximum number of timesteps to take in the env per episode.
        obs_mode: The mode for the obseravtion. Options:
            sim: Returns an observation that allows for simulating the mujoco simulator (default).
            original: Returns the original observation from the environment.
        terminate_when_unhealthy: If True, terminates the episode when healthy state bounds are exceeded.
        healthy_velocity_range: Tuple of min/max velocity values that define healthy bounds.
            These exist because without them rllib sometimes errors out with nan gradients when there are
            very large velocity values.
    """
    def __init__(
            self,
            frame_skip: int = 5,
            max_timesteps: int = 100,
            obs_mode: str = "sim",
            terminate_when_unhealthy: bool = True,
            healthy_velocity_range: Tuple[int, int] = (-50, 50),
    ):
        # These have to be stored before super init b/c it calls step.
        self.max_timesteps = max_timesteps
        self.t = 0
        self.obs_mode = obs_mode
        self.terminate_when_unhealthy = terminate_when_unhealthy
        self.healthy_velocity_range = healthy_velocity_range

        super().__init__()

        # Overwrite frame skip after calling super init.
        self.frame_skip = frame_skip
        self.metadata["video.frames_per_second"] = int(np.round(1.0 / self.dt))

    def is_healthy(self) -> bool:
        """Returns True if the simulator is in a healthy state."""
        min_velocity, max_velocity = self.healthy_velocity_range
        velocity = self.sim.data.qvel.flat[:]
        healthy_velocity = np.all(np.logical_and(min_velocity < velocity, velocity < max_velocity))

        healthy = healthy_velocity
        return healthy

    def reset(self) -> np.ndarray:
        """Resets the environment."""
        self.t = 0
        return super().reset()

    def step(self, *args, **kwargs) -> Tuple:
        """Fixes a non-json-writable element in the info of the base env."""
        obs, reward, done, info = super().step(*args, **kwargs)
        info["reward_ctrl"] = float(info["reward_ctrl"])

        if self.terminate_when_unhealthy and not self.is_healthy():
            done = True

        self.t += 1
        if self.t >= self.max_timesteps:
            done = True
        return obs, reward, done, info

    def _get_obs(self) -> np.ndarray:
        """Optionally overwrite the observation to for simulation purposes."""
        if self.obs_mode == "sim":
            return np.concatenate([
                self.sim.data.qpos.flat[:],
                self.sim.data.qvel.flat[:],
                self.get_body_com("fingertip") - self.get_body_com("target"),
            ])
        elif self.obs_mode == "original":
            return super()._get_obs()
        else:
            raise ValueError(f"Invalid observation mode: {self.obs_mode}")


class CustomReacherEnvRewardModel(RewardModel):
    """Reward model for custom Reacher environment.

    Args:
        obs_space: The observation space used in the environment.
        act_space: The action space used in the environment.
        reward_dist_factor: Weight on the distance from goal reward term.
        reward_ctrl_factor: Weight on the control reward term.
        reward_goal_factor: Weight on reaching the goal.
        shaping_factor: The value to scale the shaping.
        shaping_discount: The discount factor used in potential shaping.
    """
    # At this threshold around 2% of initial states are next to the goal.
    GOAL_REACHED_THRESHOLD = 0.05

    def __init__(
            self,
            obs_space: gym.spaces.Space,
            act_space: gym.spaces.Space,
            reward_dist_factor: float,
            reward_ctrl_factor: float,
            reward_goal_factor: float,
            shaping_factor: float,
            shaping_discount: float,
    ):
        self.obs_space = obs_space
        self.act_space = act_space
        self.reward_dist_factor = reward_dist_factor
        self.reward_ctrl_factor = reward_ctrl_factor
        self.reward_goal_factor = reward_goal_factor
        self.shaping_factor = shaping_factor
        self.shaping_discount = shaping_discount

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self.obs_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.act_space

    def reward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: Optional[torch.Tensor],
            terminals: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Computes the reward for the environment.

        See base class for documentation on args and return value.
        """
        del terminals
        states_dists = states[:, -3:].norm(dim=-1, keepdim=True)
        dist_rewards = -states_dists
        ctrl_rewards = -actions.square().sum(dim=1, keepdim=True).to(states.dtype)
        goal_rewards = states_dists < self.GOAL_REACHED_THRESHOLD

        next_states_dists = next_states[:, -3:].norm(dim=-1, keepdim=True)
        shaping_rewards = (self.shaping_discount * next_states_dists - states_dists)

        rewards = self.reward_dist_factor * dist_rewards \
            + self.reward_ctrl_factor * ctrl_rewards \
            + self.reward_goal_factor * goal_rewards \
            + self.shaping_factor * shaping_rewards

        return rewards


# %%
from typing import Optional, Tuple

import gym
from gym.envs.mujoco.reacher import ReacherEnv
import numpy as np
import torch
import mujoco_py


class CustomReacherEnv(ReacherEnv):
    """A customized version of the reacher env.

    Customization includes frame skip, changing the obs to allow for simulation from it,
    making the info dict json serializable, setting a finite horizon independent of the
    gym wrapper for doing so, and other changes.

    Args:
        frame_skip: Number of frames to skip between timesteps.
        max_timesteps: The maximum number of timesteps to take in the env per episode.
        obs_mode: The mode for the obseravtion. Options:
            sim: Returns an observation that allows for simulating the mujoco simulator (default).
            original: Returns the original observation from the environment.
        terminate_when_unhealthy: If True, terminates the episode when healthy state bounds are exceeded.
        healthy_velocity_range: Tuple of min/max velocity values that define healthy bounds.
            These exist because without them rllib sometimes errors out with nan gradients when there are
            very large velocity values.
    """
    def __init__(
            self,
            frame_skip: int = 5,
            max_timesteps: int = 100,
            obs_mode: str = "sim",
            terminate_when_unhealthy: bool = True,
            healthy_velocity_range: Tuple[int, int] = (-50, 50),
    ):
        # These have to be stored before super init b/c it calls step.
        self.max_timesteps = max_timesteps
        self.t = 0
        self.obs_mode = obs_mode
        self.terminate_when_unhealthy = terminate_when_unhealthy
        self.healthy_velocity_range = healthy_velocity_range

        super().__init__()

        # Overwrite frame skip after calling super init.
        self.frame_skip = frame_skip
        self.metadata["video.frames_per_second"] = int(np.round(1.0 / self.dt))

    def is_healthy(self) -> bool:
        """Returns True if the simulator is in a healthy state."""
        min_velocity, max_velocity = self.healthy_velocity_range
        velocity = self.sim.data.qvel.flat[:]
        healthy_velocity = np.all(np.logical_and(min_velocity < velocity, velocity < max_velocity))

        healthy = healthy_velocity
        return healthy

    def reset(self) -> np.ndarray:
        """Resets the environment."""
        self.t = 0
        return super().reset()

    def step(self, *args, **kwargs) -> Tuple:
        """Fixes a non-json-writable element in the info of the base env."""
        obs, reward, done, info = super().step(*args, **kwargs)
        info["reward_ctrl"] = float(info["reward_ctrl"])

        if self.terminate_when_unhealthy and not self.is_healthy():
            done = True

        self.t += 1
        if self.t >= self.max_timesteps:
            done = True
        return obs, reward, done, info

    def _get_obs(self) -> np.ndarray:
        """Optionally overwrite the observation to for simulation purposes."""
        if self.obs_mode == "sim":
            return np.concatenate([
                self.sim.data.qpos.flat[:],
                self.sim.data.qvel.flat[:],
                self.get_body_com("fingertip") - self.get_body_com("target"),
            ])
        elif self.obs_mode == "original":
            return super()._get_obs()
        else:
            raise ValueError(f"Invalid observation mode: {self.obs_mode}")


class CustomReacherEnvRewardModel(RewardModel):
    """Reward model for custom Reacher environment.

    Args:
        obs_space: The observation space used in the environment.
        act_space: The action space used in the environment.
        reward_dist_factor: Weight on the distance from goal reward term.
        reward_ctrl_factor: Weight on the control reward term.
        reward_goal_factor: Weight on reaching the goal.
        shaping_factor: The value to scale the shaping.
        shaping_discount: The discount factor used in potential shaping.
    """
    # At this threshold around 2% of initial states are next to the goal.
    GOAL_REACHED_THRESHOLD = 0.05

    def __init__(
            self,
            obs_space: gym.spaces.Space,
            act_space: gym.spaces.Space,
            reward_dist_factor: float,
            reward_ctrl_factor: float,
            reward_goal_factor: float,
            shaping_factor: float,
            shaping_discount: float,
    ):
        self.obs_space = obs_space
        self.act_space = act_space
        self.reward_dist_factor = reward_dist_factor
        self.reward_ctrl_factor = reward_ctrl_factor
        self.reward_goal_factor = reward_goal_factor
        self.shaping_factor = shaping_factor
        self.shaping_discount = shaping_discount

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self.obs_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.act_space

    def reward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: Optional[torch.Tensor],
            terminals: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Computes the reward for the environment.

        See base class for documentation on args and return value.
        """
        del terminals
        states_dists = states[:, -3:].norm(dim=-1, keepdim=True)
        dist_rewards = -states_dists
        ctrl_rewards = -actions.square().sum(dim=1, keepdim=True).to(states.dtype)
        goal_rewards = states_dists < self.GOAL_REACHED_THRESHOLD

        next_states_dists = next_states[:, -3:].norm(dim=-1, keepdim=True)
        shaping_rewards = (self.shaping_discount * next_states_dists - states_dists)

        rewards = self.reward_dist_factor * dist_rewards \
            + self.reward_ctrl_factor * ctrl_rewards \
            + self.reward_goal_factor * goal_rewards \
            + self.shaping_factor * shaping_rewards

        return rewards


# %%


env = CustomReacherEnv(obs_mode='original')

num_episodes = 10

num_steps = 10

observations = []
actions = []
rewards = []
infos = []

for episode in range(num_episodes):
    observation = env.reset()
    observations.append([observation]) 
    episode_rewards = []
    for _ in range(num_steps):
        # Random action using the action space of CustomReacherEnv
        action = env.action_space.sample()
        actions.append(action)
        
        observation, reward, done, info = env.step(action)
        observations[-1].append(observation)  # Append new observation to the current episode
        episode_rewards.append(reward)
        infos.append(info)

        if done:
            break

    rewards.append(episode_rewards)  # Append episode rewards

env.close()

# Now, you can print or analyze your values
print("Actions:", actions)
print("Rewards:", rewards)
# And so on for other values...


# %%
import stable_baselines3

# %%
import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import PPO, MlpPolicy

from stable_baselines3.common.vec_env import DummyVecEnv

# Wrap your custom environment. VecEnvs are typically used for better performance.
env = DummyVecEnv([lambda: CustomReacherEnv(obs_mode='original')])


# Instantiate the agent
model = PPO(MlpPolicy, env, verbose=1)
# Train the agent
model.learn(total_timesteps=2000)


# %%
import numpy as np

# Number of episodes
num_episodes = 10

# Number of steps per episode
num_steps = 200

for episode in range(num_episodes):
    obs = env.reset()
    total_reward = 0
    for _ in range(num_steps):
        # Predict the action using the trained policy
        action, _ = model.predict(obs)
        
        # Step through the environment
        obs, reward, done, info = env.step(action)
        
        # Accumulate the reward
        total_reward += reward
        
        if done:
            break
    
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()


# %%
def predict_next_state(inner_env, state, action):
    # Set environment to desired state
    inner_env.set_state(state[:inner_env.model.nq], state[inner_env.model.nq:inner_env.model.nq + inner_env.model.nv])
    
    # Take the desired action
    next_state, _, _, _ = inner_env.step(action)
    
    return next_state

# Extract the inner environment from DummyVecEnv
inner_env = env.envs[0]

# Example usage:
current_state = env.reset()
action, _ = model.predict(current_state)

predicted_next_state = predict_next_state(inner_env, current_state[0], action)
print(predicted_next_state)


# %%
def predict_next_state(inner_env, state, action):
    # Set environment to desired state
    inner_env.set_state(state[:inner_env.model.nq], state[inner_env.model.nq:inner_env.model.nq + inner_env.model.nv])
    
    # Take the desired action
    next_state, _, _, _ = inner_env.step(action)
    
    return next_state

# Extract the inner environment from DummyVecEnv
inner_env = env.envs[0]

# Initialize current state
current_state = env.reset()

# Initialize tracking variables
max_zeroth_index = current_state[0][0]
min_zeroth_index = current_state[0][0]
cumulative_sum = current_state[0][0]

# Go through 1000 steps
for i in range(1000):
    # Predict action based on the current state
    action, _ = model.predict(current_state)
    
    # Predict the next state based on the current state and action
    predicted_next_state = predict_next_state(inner_env, current_state[0], action)
    
    # Update the max, min, and cumulative sum for the 0th index
    max_zeroth_index = max(max_zeroth_index, predicted_next_state[0])
    if max_zeroth_index>1:
        print(predicted_next_state)
        print("YO")
    min_zeroth_index = min(min_zeroth_index, predicted_next_state[0])
    cumulative_sum += predicted_next_state[0]
    
    # Update the current state for the next iteration
    current_state[0] = predicted_next_state

# Calculate the average value for the 0th index
average_zeroth_index = cumulative_sum / 1001  # Including the initial state

# Print the results
print("Max value of the 0th index:", max_zeroth_index)
print("Min value of the 0th index:", min_zeroth_index)
print("Average value of the 0th index:", average_zeroth_index)


# %%
def predict_next_state(inner_env, state, action):
    inner_env.set_state(state[:inner_env.model.nq], state[inner_env.model.nq:inner_env.model.nq + inner_env.model.nv])
    

    next_state, _, _, _ = inner_env.step(action)
    
    return next_state

inner_env = env.envs[0]

current_state = env.reset()
print(f"Current state: {current_state[0]}")

action, _ = model.predict(current_state)

predicted_next_state = predict_next_state(inner_env, current_state[0], action)
print(f"Predicted next state: {predicted_next_state}")


# %%
def predict_next_state(inner_env, state, action):
    inner_env.set_state(state[:inner_env.model.nq], state[inner_env.model.nq:inner_env.model.nq + inner_env.model.nv])
    next_state, _, _, _ = inner_env.step(action)
    return next_state

inner_env = env.envs[0]

# Initialize current state
current_state = env.reset()


max_values = [-float('inf') for _ in current_state[0]]

for _ in range(100000):
    action, _ = model.predict(current_state)
    predicted_next_state = predict_next_state(inner_env, current_state[0], action)

    max_values = [max(max_val, state_val) for max_val, state_val in zip(max_values, predicted_next_state)]
    current_state[0] = predicted_next_state


for idx, max_val in enumerate(max_values):
    print(f"Index {idx}: Max Value = {max_val}")


# %%
def predict_next_state(inner_env, state, action):
    inner_env.set_state(state[:inner_env.model.nq], state[inner_env.model.nq:inner_env.model.nq + inner_env.model.nv])
    next_state, _, _, _ = inner_env.step(action)
    return next_state

inner_env = env.envs[0]

# Initialize current state
current_state = env.reset()


max_values = [-float('inf') for _ in current_state[0]]

for _ in range(100000):
    action, _ = model.predict(current_state)
    predicted_next_state = predict_next_state(inner_env, current_state[0], action)

    max_values = [max(max_val, state_val) for max_val, state_val in zip(max_values, predicted_next_state)]
    current_state[0] = predicted_next_state


for idx, max_val in enumerate(max_values):
    print(f"Index {idx}: Max Value = {max_val}")


# %%
obs

# %%
len(obs[0])

# %%
#Values for states
obs = [0.08370445, -0.28718871, -0.07374638, -0.15162033, 7.01024685, -8.4554024, 0., 0., 0.28112693, 0.13775099, 0.]
obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)

# Predict the value
value_tensor = model.policy.predict_values(obs_tensor)

# Convert the tensor to a Python number
value = value_tensor.item()
print("Value of the given state:", value)


# %%
from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")


# %%
obs

# %%
print(env.observation_space.shape)


# %%



