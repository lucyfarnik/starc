import numpy as np

# just here for readability in the other files
Reward = np.ndarray # [S, A, S']; reward at transition s -a-> s'
Policy = np.ndarray # [S] action to take in state; deterministic
