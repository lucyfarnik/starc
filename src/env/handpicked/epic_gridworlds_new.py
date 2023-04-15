# taken from the EPIC paper section 5
import numpy as np
from env import Env, RandomEnv
from _types import Reward

"""Illustrative rewards for gridworlds."""

import numpy as np
def epic_gridworlds():
    SPARSE_GOAL = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])

    CENTER_GOAL = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    OBSTACLE_COURSE = np.array([[0, -1, -1], [0, 0, 0], [-1, -1, 4]])

    CLIFF_WALK = np.array([[0, -1, -1], [0, 0, 0], [-4, -4, 4]])

    MANHATTAN_FROM_GOAL = np.array([[4, 3, 2], [3, 2, 1], [2, 1, 0]])

    ZERO = np.zeros((3, 3))

    REWARDS = {
        # Equivalent rewards
        "sparse_goal": {"state_reward": SPARSE_GOAL, "potential": ZERO},
        "sparse_goal_shift": {"state_reward": SPARSE_GOAL + 1, "potential": ZERO},
        "sparse_goal_scale": {"state_reward": SPARSE_GOAL * 10, "potential": ZERO},
        "dense_goal": {"state_reward": SPARSE_GOAL, "potential": -MANHATTAN_FROM_GOAL},
        "antidense_goal": {"state_reward": SPARSE_GOAL, "potential": MANHATTAN_FROM_GOAL},
        # Non-equivalent rewards
        "transformed_goal": {
            # Shifted, rescaled and reshaped sparse goal.
            "state_reward": SPARSE_GOAL * 4 - 1,
            "potential": -MANHATTAN_FROM_GOAL * 3,
        },
        "center_goal": {
            # Goal is in center
            "state_reward": CENTER_GOAL,
            "potential": ZERO,
        },
        "dirt_path": {
            # Some minor penalties to avoid to reach goal.
            #
            # Optimal policy for this is optimal in `SPARSE_GOAL`, but not equivalent.
            # Think may come apart in some dynamics but not particularly intuitively.
            "state_reward": OBSTACLE_COURSE,
            "potential": ZERO,
        },
        "cliff_walk": {
            # Avoid cliff to reach goal. Same set of optimal policies as `obstacle_course` in
            # deterministic dynamics, but not equivalent.
            #
            # Optimal policy differs in sufficiently slippery gridworlds as want to stay on top line
            # to avoid chance of falling off cliff.
            "state_reward": CLIFF_WALK,
            "potential": ZERO,
        },
        "sparse_penalty": {
            # Negative of `sparse_goal`.
            "state_reward": -SPARSE_GOAL,
            "potential": ZERO,
        },
        "evaluating_rewards/Zero-v0": {
            # All zero reward function
            "state_reward": ZERO,
            "potential": ZERO,
        },
    }

    sparse_var = REWARDS["sparse_goal"]["state_reward"]
    dense_var = REWARDS["dense_goal"]["state_reward"]
    cliff_var = REWARDS["cliff_walk"]["state_reward"]
    path_var = REWARDS["dirt_path"]["state_reward"]

    cliff_var_f = cliff_var.flatten()
    dense_var_f = dense_var.flatten()
    sparse_var_f = sparse_var.flatten()
    path_var_f = path_var.flatten()

    sparse_reward = np.zeros((9,5,9)) + sparse_var_f[:,None,None]
    dense_reward = np.zeros((9,5,9)) + dense_var_f[:,None,None]
    path_reward = np.zeros((9,5,9)) + path_var_f[:,None,None]
    cliff_reward = np.zeros((9,5,9)) + cliff_var_f[:,None,None]

    r = {'sparse': sparse_reward,
        'dense': dense_reward,
        'path': path_reward,
        'cliff': cliff_reward}

    env = RandomEnv(n_s=9, n_a=5, discount=0.99) 
    return env, r
