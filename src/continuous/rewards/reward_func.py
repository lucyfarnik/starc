import abc
from typing import Optional
import torch
from continuous.env import ReacherEnv

class RewardFunc(abc.ABC):
    @abc.abstractmethod
    def __call__(
            self,
            env: ReacherEnv,
            state: Optional[torch.Tensor], #TODO fix the types
            action,
            next_state) -> float:
