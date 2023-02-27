import jax.numpy as jnp
from env import Env

# for the tests we'll use n_s=2, n_a=2, with deterministic transitions and rewards
# action 0 always keeps you in the current state, action 1 switches to the other
# rewards are 0 on all transitions, except for s0 -> s1 where r=1
# this lets us work out the correct Q values by hand as:
# Q(1, b) = 1 / (1-gamma^2)
# Q(1, a) = Q(2, b) = gamma / (1-gamma^2)
# Q(2, a) = gamma^2 / (1-gamma^2)
n_s = 2
n_a = 2
discount = 0.9
init_dist = jnp.array([0.5, 0.5])
transition_dist = jnp.array([
  [ # s=0
    [ # a=0
      1.0, # s_next = 0
      0.0, 
    ],
    [ # a=1
      0.0,
      1.0,
    ],
  ],
  [ # s=1
    [ # a=0
      0.0,
      1.0,
    ],
    [ # a=1
      1.0,
      0.0,
    ]
  ]
])
env = Env(n_s, n_a, discount, init_dist, transition_dist)

reward = jnp.zeros((n_s, n_a, n_s))
reward = reward.at[0, 1, 1].set(1.0)

expected_q_vals = jnp.array([
  [ # s=0
    discount / (1-discount**2),
    1 / (1-discount**2),
  ],
  [ # s=1
    discount**2 / (1-discount**2),
    discount / (1-discount**2),
  ]
])

expected_policy = expected_q_vals.argmax(axis=1)
