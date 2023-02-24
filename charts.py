# work in progress
import numpy as np
import matplotlib.pyplot as plt
import json

results = None
with open('results.json', 'r') as f:
  results = json.load(f)[0]

flat = [i for r in results for i in r]

x_axis = np.arange(len(results[0]))

fig, ax = plt.subplots(ncols=3)
for i, result in enumerate(results):
  for key in result[0].keys():
    if 'regret' in key or key == 'EPIC-infty-1' or key == 'EPIC-2-1':
      continue
    ax[i].plot(x_axis, [r[key] for r in result], label=key)
  ax[i].legend()

plt.show()

# for every env, graph pairs of different measurements between reward functions
# eg EPIC vs DARD, with a dot for every reward function