import numpy as np
import matplotlib.pyplot as plt
import json

results = None
with open('results/5_consistent_rollouts.json', 'r') as f:
  results = json.load(f)

num_envs = len(results)
num_rs = len(results[0])
num_interp_steps = len(results[0][0])

x_axis = np.arange(num_interp_steps)
keys = results[0][0][0].keys()

fig, ax = plt.subplots(nrows=num_envs, ncols=num_rs)
for env_i, env_data in enumerate(results):
  for r_i, r_data in enumerate(env_data):
    for key in keys:
      ax[env_i][r_i].scatter(x_axis, [r[key] for r in r_data],
                             label=key, marker='.')
    ax[env_i][r_i].set_yscale('log')

fig.suptitle('Consistent rollout values')
fig.legend(keys)
fig.supxlabel('Reward samples')
fig.supylabel('Environment samples')

plt.show()

# for every env, graph pairs of different measurements between reward functions
# eg EPIC vs DARD, with a dot for every reward function

# TODO: pairplot everything compared to EPIC