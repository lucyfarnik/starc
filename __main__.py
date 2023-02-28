import numpy as np
import json
from env import RandomEnv
from reward import random_reward, interpolate
from canon import canon_and_norm
from policy import optimize, policy_return, policy_returns
from env import RandomEnv

# hyperparams
num_trials = 4
num_samples = 4
interpolation_steps = 32
n_s = 32
n_a = 8

dist_opts = [1, 2, float('inf')]

# experiment starts here
results = []
for trial_i in range(num_trials): # each trial comes with its own environment
  trail_results = []
  env = RandomEnv(n_s, n_a)

  for sample_i in range(num_samples): # sampling different rewards
    print(f"Environment {trial_i}, rewards {sample_i}")
    sample_results = []

    # generate random rewards
    r1, r2 = random_reward(env), random_reward(env)

    # policies for R1
    pi_1 = optimize(env, r1) # best policy under R1
    pi_x = optimize(env, -r1) # worst policy under R1

    # return values under R1
    J_1_pi_1 = policy_return(r1, pi_1, env)
    J_1_pi_x = policy_return(r1, pi_x, env)

    # canonicalizations and normalizations of R1
    can1 = canon_and_norm(r1, env)

    # interpolate between R1 and R2
    for r_i in interpolate(r1, r2, interpolation_steps):
      interp_results = {}

      # canonicalizations and normalizations of R_i
      can_i = canon_and_norm(r_i, env)

      # compute the distances for all combinations of canon, norm, and dist
      for cn_name, r1_val in can1.items():
        r_i_val = can_i[cn_name]
        for d_ord in dist_opts:
          interp_results[f'{cn_name}-{d_ord}'] = np.linalg.norm(
            (r1_val - r_i_val).flatten(), d_ord
          )

      # policies for R_i
      pi_i = optimize(env, r_i) # best policy under R_i
      pi_y = optimize(env, -r_i) # worst policy under R_i

      # compute return values
      J_i_pi_1 = policy_return(r_i, pi_1, env)
      J_1_pi_i, J_i_pi_i = policy_returns([r1, r_i], pi_i, env)
      J_i_pi_y = policy_return(r_i, pi_y, env)
      #! Check the values here, make sure the policy roll out values do not overlap with the distance between J_1_pi_1 and J_1_pi_i
      #! somehow J_i_pi_1 is often larger than J_i_pi_i which makes no sense
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
