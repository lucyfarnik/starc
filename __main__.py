from jax import random
import json
from env import RandomEnv
from reward import random_reward, interpolate
from policy import optimize, policy_return
from distance import RewardDistance
from env import RandomEnv

# hyperparams
num_trials = 4
num_samples = 4
interpolation_steps = 64

rand_key = random.PRNGKey(42)

distances = {}
# add all mixes of norms for norm and dist functions, instantiate distances for them
canon_opts = ['EPIC', 'DARD']
norm_opts = [1, 2, float('inf')]
for c in canon_opts:
  for n in norm_opts:
    for d in norm_opts:
      distances[f'{c}-{n}-{d}'] = RewardDistance(c, n, d)

# experiment starts here
results = []
for trial_i in range(num_trials): # each trial comes with its own environment
  trail_results = []
  rand_key, *rand_subkeys = random.split(rand_key, 3)
  env = RandomEnv(rand_subkeys)

  for sample_i in range(num_samples): # sampling different rewards
    print(f"Trial {trial_i}, sample {sample_i}")
    sample_results = []

    # generate random rewards
    rand_key, rand_subkey1, rand_subkey2 = random.split(rand_key, 3)
    r1, r2 = random_reward(env, rand_subkey1), random_reward(env, rand_subkey2)

    # policies for R1
    pi_1, rand_key = optimize(env, r1, rand_key) # best policy under R1
    pi_x, rand_key = optimize(env, -r1, rand_key) # worst policy under R1

    # interpolate between R1 and R2
    for r_i in interpolate(r1, r2, interpolation_steps):
      interp_results = {}

      # compute distance values between R1 and R_i
      # TODO: optimize, this currently computes identical canonicalization 8 times
      for dist_name, dist in distances.items():
        interp_results[dist_name] = dist(r1, r_i, env)

      # policies for R_i
      pi_i, rand_key = optimize(env, r_i, rand_key) # best policy under R_i
      pi_y, rand_key = optimize(env, -r_i, rand_key) # worst policy under R_i

      # compute return values
      # TODO: optimize this to avoid duplicate calculation (change the func to allow multiple reward functions)
      J_1_pi_1, rand_key = policy_return(r1, pi_1, env, rand_key)
      J_1_pi_i, rand_key = policy_return(r1, pi_i, env, rand_key)
      J_1_pi_x, rand_key = policy_return(r1, pi_x, env, rand_key)
      J_i_pi_1, rand_key = policy_return(r_i, pi_1, env, rand_key)
      J_i_pi_i, rand_key = policy_return(r_i, pi_i, env, rand_key)
      J_i_pi_y, rand_key = policy_return(r_i, pi_y, env, rand_key)
      #! Check the values here, make sure the policy roll out values do not overlap with the distance between J_1_pi_1 and J_1_pi_i
      # compute regrets (normalizing over max-min)
      interp_results['regret1'] = (J_1_pi_1 - J_1_pi_i) / (J_1_pi_1 - J_1_pi_x)
      interp_results['regret2'] = (J_i_pi_i - J_i_pi_1) / (J_i_pi_i - J_i_pi_y)

      # save results
      sample_results.append(interp_results)
    trail_results.append(sample_results)
  results.append(trail_results)
        
# save as JSON
with open('results.json', 'w') as f:
  json.dump(results, f)
