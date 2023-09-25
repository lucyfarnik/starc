from typing import Tuple
from gym.envs.mujoco.reacher import ReacherEnv as OriginalReacher
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import PPO, MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

class ReacherEnv(OriginalReacher):
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
    # Set environment to desired state
    env.set_state(state[:env.model.nq], state[env.model.nq:env.model.nq + env.model.nv])
    
    # Take the desired action
    next_state, _, _, _ = env.step(action)
    
    return next_state
