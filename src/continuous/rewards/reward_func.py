import abc

class RewardFunc(abc.ABC):
    @abc.abstractmethod
    def __call__(self, env, state, action, next_state) -> float:
        raise NotImplementedError
