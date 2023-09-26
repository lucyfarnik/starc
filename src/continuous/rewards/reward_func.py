import abc
from typing import Optional
import torch
from continuous.env import ReacherEnv

class RewardFunc(abc.ABC):
    def __init__(self, env: ReacherEnv):
        self.env = env

    @abc.abstractmethod
    def __call__(
            self,
            state: Optional[torch.Tensor], #TODO fix the types
            action,
            next_state) -> float:
