"""
Functions to compute gradient, divergence, and Laplacian,
as well as divergence-free representatives.

Setting is tabular with R(s, a, s') reward functions.
A potential is represented as a numpy array of length |S|,
and a reward functions as an array with shape (|S|, |A|, |S|),
where |S| is the number of states and |A| is the number of actions.

Axes before that are treated as batch axes.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class TransitionGraph(ABC):
    def __init__(self, n_states, n_actions, gamma, weights=None):
        self.gamma = gamma
        self.n_states = n_states
        self.n_actions = n_actions
        self.states = np.arange(n_states)
        self.actions = np.arange(n_actions)
        self.weights = weights
        if self.weights is None:
            self.weights = np.ones(self.reward_shape) / self.num_transitions
        if np.any(self.weights < 0):
            raise ValueError("Weights must be non-negative.")
        # Note we don't want to do this in-place, since that would modify the input
        self.weights = self.weights / np.sum(self.weights)

    @property
    @abstractmethod
    def reward_shape(self) -> Tuple[int, ...]:
        """Return the shape of a reward function array on this graph."""

    @property
    def num_transitions(self) -> int:
        """The number of total possible transitions (edges in the graph)."""
        return np.product(self.reward_shape)

    @abstractmethod
    def gradient(self, potential: np.ndarray) -> np.ndarray:
        """Compute the discounted gradient of a given potential."""

    @abstractmethod
    def inflow(self, rewards: np.ndarray) -> np.ndarray:
        """Compute the inflow of a given reward function at each state."""

    @abstractmethod
    def outflow(self, rewards: np.ndarray) -> np.ndarray:
        """Compute the outflow of a given reward function at each state."""

    def divergence(self, rewards):
        """Compute the divergence of a given reward function."""
        return self.gamma * self.inflow(rewards) - self.outflow(rewards)

    def laplacian(self, potential: np.ndarray) -> np.ndarray:
        """Compute the Laplacian of a given potential."""
        return self.divergence(self.gradient(potential))

    def laplacian_matrix(self) -> np.ndarray:
        """Compute the matrix representation of the Laplacian operator."""
        potential = np.eye(self.n_states)
        return self.laplacian(potential).T

    def outflow_of_gradient_matrix(self) -> np.ndarray:
        """Compute the matrix of outflow composed with gradient."""
        return self.outflow(self.gradient(np.eye(self.n_states))).T

    def divergence_free(self, rewards: np.ndarray) -> np.ndarray:
        """Compute the divergence-free representation of a given reward function."""
        matrix = self.laplacian_matrix()
        div = self.divergence(rewards)
        potential = np.linalg.solve(matrix, div)
        return rewards - self.gradient(potential)

    def outflow_free(self, rewards: np.ndarray) -> np.ndarray:
        """Compute the outflow-free representation of a given reward function."""
        matrix = self.outflow_of_gradient_matrix()
        outflow = self.outflow(rewards)
        potential = np.linalg.solve(matrix, outflow)
        return rewards - self.gradient(potential)

    def norm(self, rewards: np.ndarray) -> np.ndarray:
        """Compute the norm of a given reward function."""
        # We want to sum only over the last len(reward_shape) axes:
        axis = tuple(range(-1, -len(self.reward_shape) - 1, -1))
        return np.sqrt(np.sum(self._weighted(rewards) ** 2, axis=axis))

    def _weighted(self, rewards: np.ndarray) -> np.ndarray:
        """Return rewards * weights."""
        # if the weights array has fewer batch dimensions, we prepend some
        weights = self.weights[(None,) * (rewards.ndim - self.weights.ndim)]
        return rewards * weights


class CompleteTransitionGraph(TransitionGraph):
    def gradient(self, potential):
        """Compute the discounted gradient of a given potential.

        The result will have a singleton axis for the action dimension.
        """
        return (
            self.gamma * potential[..., None, None, :] - potential[..., :, None, None]
        )

    def inflow(self, rewards: np.ndarray) -> np.ndarray:
        """Compute the inflow of a given reward function at each state."""
        weighted_rewards = self._weighted(rewards)
        return weighted_rewards.sum((-3, -2))

    def outflow(self, rewards: np.ndarray) -> np.ndarray:
        """Compute the outflow of a given reward function at each state."""
        weighted_rewards = self._weighted(rewards)
        return weighted_rewards.sum((-2, -1))

    @property
    def reward_shape(self) -> Tuple[int, int, int]:
        return (self.n_states, self.n_actions, self.n_states)

    def epic_shaped(self, rewards: np.ndarray) -> np.ndarray:
        """Compute the canonicalization of a given reward function used by EPIC.

        Note that EPIC requires a complete graph, so this is only implemented here.
        """
        # mean_rew_s = np.average(rewards, axis=(1, 2))
        # mean_rew = np.average(mean_rew_s, axis=0)
        state_dist = np.sum(self.weights, axis=(0, 1), keepdims=True)
        action_dist = np.sum(self.weights, axis=(0, 2), keepdims=True)
        potential = (rewards * action_dist * state_dist).sum(axis=(1, 2))
        avg_reward = (potential * state_dist.flatten()).sum()
        return rewards + self.gradient(potential) - self.gamma * avg_reward

    def dard_shaped(self, rewards: np.ndarray) -> np.ndarray:
        """Compute the DARD transformation of a given reward function."""
        # outflow contains state distribution, which we want to remove:
        potential = self.outflow(rewards) / self.weights.sum(axis=(1, 2))
        joint_state_probs = self.weights.sum(axis=1)
        # next_state_probs[s, s'] = P(s' | s)
        next_state_probs = joint_state_probs / joint_state_probs.sum(
            axis=1, keepdims=True
        )
        action_probs = self.weights.sum(axis=(0, 2))
        # I'm calling this an "offset" in analogy to EPIC, but note that it's not
        # a constant!
        offset = (
            rewards[None, :, :, None, :]
            * next_state_probs[:, :, None, None, None]
            * action_probs[None, None, :, None, None]
            * next_state_probs[None, None, None, :, :]
        ).sum(axis=(1, 2, 4))

        return rewards + self.gradient(potential) - self.gamma * offset[:, None, :]

    def dard_matrix(self) -> np.ndarray:
        """Compute the matrix representation of the DARD transformation."""
        # dard_shaped currently doesn't support batching, so we need to do it manually:
        matrix = np.zeros((self.num_transitions,) * 2)
        for i in range(self.num_transitions):
            reward = np.zeros(self.num_transitions)
            reward[i] = 1
            matrix[:, i] = self.dard_shaped(reward.reshape(self.reward_shape)).flatten()
        return matrix


class DeterministicTransitionGraph(TransitionGraph):
    """Efficient implementation of the transition graph for deterministic MDPs.

    The difference to complete transition graphs is that we represent rewards
    as |S| x |A| instead of |S| x |A| x |S| arrays.

    Note that this class assumes each action can be taken in each state.
    """

    def __init__(self, next_states: np.ndarray, gamma: float, weights=None):
        """Initialize a DeterministicTransitionGraph.

        Args:
            next_states: An array of shape (|S|, |A|) containing the transition
                dynamics (i.e. next_states[s, a] is the state that is reached
                by taking action `a` in state `s`). May be a masked array,
                where the masked entries are invalid transitions (in case some
                action is not available in all states).
            gamma: discount factor
            weights: Optional array of shape (|S|, |A|) containing the weights
                of transitions.
        """
        n_states, n_actions = next_states.shape
        if np.ma.is_masked(next_states) and weights is not None:
            weights = np.ma.asarray(weights)
            weights.mask = next_states.mask
        super().__init__(n_states, n_actions, gamma, weights=weights)
        self.next_states = next_states

    def in_transitions(self, state: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get all the transitions into `state`.

        Returns a a tuple of two arrays, the first containing the states
        and the second containing the corresponding actions of incoming transitions.

        This method uses `DeterministicTransitionGraph.next_states` to compute
        these incoming transitions. Subclasses may want to override this method
        with a more efficient implementation.

        The reason this function does not accept an array of states is that
        each state may have a different number of transitions into it, making
        batch processing difficult.
        """
        return np.nonzero(self.next_states == state)

    def inflow(self, rewards: np.ndarray) -> np.ndarray:
        """Compute the inflow of a given reward function at each state.

        In graphs where each node has the same number of incoming transitions,
        you may want to override this implementation with a more efficient
        vectorized version (rather than calling `in_transitions` for each state).
        """
        weighted_rewards = self._weighted(rewards)
        batch_dims = weighted_rewards.shape[:-2]

        inflow = np.zeros(batch_dims + (self.n_states,))
        for i in range(self.n_states):
            in_states, in_actions = self.in_transitions(i)
            inflow[..., i] = weighted_rewards[..., in_states, in_actions].sum(-1)

        return inflow

    def outflow(self, rewards: np.ndarray) -> np.ndarray:
        """Compute the outflow of a given reward function at each state."""
        weighted_rewards = self._weighted(rewards)
        # sum over only the action:
        return weighted_rewards.sum(-1)

    def gradient(self, potential):
        """Compute the discounted gradient of a given potential."""
        # The current potential is broadcast over the action dimension
        return self.gamma * potential[..., self.next_states] - potential[..., :, None]

    @property
    def reward_shape(self) -> Tuple[int, int]:
        return (self.n_states, self.n_actions)

    def possible_transitions(self) -> np.ndarray:
        """Return an |S| x |A| x |S| boolean array which is True iff the transition is possible.

        Requires masked values in `next_states` to be valid indices (even though they're
        not used). TODO: should probably fix that."""
        out = np.zeros((self.n_states, self.n_actions, self.n_states), dtype=bool)
        out[self.states[:, None], self.actions[None, :], self.next_states] = True
        if np.ma.is_masked(self.next_states):
            # masked transitions are not possible
            out[self.next_states.mask, :] = False
        return out

    def to_complete(self, rewards: np.ndarray) -> np.ndarray:
        """Convert a |S| x |A| array of rewards for this graph to a
        |S| x |A| x |S| array of rewards for the corresponding CompleteTransitionGraph.

        Rewards for transitions that are impossible under the deterministic dynamics
        will be set to 0.
        TODO: is that a good idea? Need to set them to something to compute EPIC though.

        WARNING: Makes use of the masked values in next_states!
        TODO: should probably set them to zero instead, but that's incompatible with
        the implementation of gridworlds in the EPIC paper.
        """
        out = np.zeros((self.n_states, self.n_actions, self.n_states))
        out[self.states[:, None], self.actions[None, :], self.next_states] = rewards
        return out


def idx2coords(idx, size):
    """Convert a linear index to a 2-tuple of coordinates for a square gridworld."""
    return (idx // size, idx % size)


def coords2idx(coords, size):
    """Convert a 2-tuple of coordinates for a square gridworld to a linear index."""
    return size * coords[0] + coords[1]


class GridworldTransitionGraph(DeterministicTransitionGraph):
    # TODO: we can provide a more efficient implementation of inflow()
    def __init__(
        self,
        size: Tuple[int, int],
        gamma: float,
        weights: Optional[np.ndarray] = None,
        include_invalid_transitions: bool = True,
        walls: Optional[np.ndarray] = None,
    ):
        """Instantiate a gridworld transition graph.

        Note that reward functions are represented as |S| x |A| arrays,
        like for DeterministicTransitionGraph (rather than having two dimensions
        to parameterize the state space). The conversion from gridworld
        coordinates (i, j) to state space indices is `state = i * size[1] + j`.

        There are five actions in each state, four directions of movement
        and one for staying in place. If an impossible movement is performed
        at the edge of the gridworld, the agent stays in place.

        Args:
            size: dimensions of the gridworld (e.g. (6, 4) gives a 6x4 gridworld)
            gamma: discount factor
            weights: optional array of weights with shape |S| x |A|
            include_invalid_transitions: whether to include (s, a) pairs into the graph
                that are not possible in the gridworld, e.g. walking out of the grid.
                If True, these invalid actions just leave the agent where it is.
            walls: optional array of shape (|S|, |A|) describing walls: if True, then the
                action can't be taken in the corresponding state. If `include_invalid_transitions`
                is set, it will still be included (and leave the state unchanged).
        """
        if include_invalid_transitions:
            next_states = np.zeros((size[0] * size[1], 5), dtype=int)
        else:
            next_states = np.ma.zeros((size[0] * size[1], 5), dtype=int)

        for i in range(size[0]):
            for j in range(size[1]):
                state = coords2idx((i, j), size[1])
                next_i, next_j = np.repeat(i, 5), np.repeat(j, 5)
                next_i[1] -= 1  # left
                next_j[2] += 1  # up
                next_i[3] += 1  # right
                next_j[4] -= 1  # down

                # actions that would take us out of the gridworld:
                invalid = np.logical_or(
                    np.logical_or(next_i < 0, next_j < 0),
                    np.logical_or(next_i >= size[0], next_j >= size[1]),
                )

                # We can't walk through walls:
                if walls is not None:
                    invalid = np.logical_or(invalid, walls[state, :])

                # invalid motions don't change the state:
                next_i[invalid] = i
                next_j[invalid] = j

                next_states[state] = coords2idx((next_i, next_j), size[1])

                if not include_invalid_transitions:
                    next_states[state, invalid] = np.ma.masked

        super().__init__(next_states=next_states, gamma=gamma, weights=weights)
