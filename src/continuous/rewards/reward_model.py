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
