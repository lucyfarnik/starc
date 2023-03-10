import math
import matplotlib.pyplot as plt
import json

results = None
with open('results/5_consistent_rollouts.json', 'r') as f:
  results = json.load(f)

# flatten
results = [r for env in results for rs in env for r in rs]
keys = [key for key in results[0].keys() if key != 'EPIC-2-2']
n_keys = len(keys)
# n_rows = math.ceil(math.sqrt(n_keys))
n_rows = 4
n_cols = math.ceil(n_keys / n_rows)

fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols)
for i, key in enumerate(keys):
  ax[i%n_rows][i//n_rows].scatter(
    [el['EPIC-2-2'] for el in results],
    [el[key] for el in results],
    label=key, marker='.')
  ax[i%n_rows][i//n_rows].set_title(key)
  ax[i%n_rows][i//n_rows].set_xscale('log')
  ax[i%n_rows][i//n_rows].set_yscale('log')

fig.suptitle('Correlations with EPIC')

plt.show()
