import numpy as np
import json
from env import RandomEnv
from reward import random_reward, interpolate
from canon import canon_and_norm
from policy import optimize, policy_return, policy_returns
from env import RandomEnv

config = {
  # hyperparams
  'num_envs': 4,
  'num_rewards': 4,
  'interpolation_steps': 16,
  'n_s': 32,
  'n_a': 4,

  # env and reward settings
  'episodic': False,
  'discount': 0.99,
  'sparse': False,
  'state_dependent': False,
  'reward_only_in_terminal': False,
}
results_path = 'results/high_discount_results.json'

dist_opts = [1, 2, float('inf')]

# experiment starts here
# TODO look at different specific envs and rewards (eg both sparse)
# TODO try to check if things get more/less correlated when we change the discount factor
results = []
for env_i in range(config['num_envs']): # each trial comes with its own environment
  trail_results = []
  env = RandomEnv(config['n_s'], config['n_a'], config['discount'], config['episodic'])

  for reward_i in range(config['num_rewards']): # sampling different rewards
    print(f"Environment {env_i}, rewards {reward_i}")
    sample_results = []

    # generate random rewards
    r1 = random_reward(env, config['sparse'], config['state_dependent'],
                       config['reward_only_in_terminal'])
    r2 = random_reward(env, config['sparse'], config['state_dependent'],
                       config['reward_only_in_terminal'])

    # policies for R1
    pi_1 = optimize(env, r1) # best policy under R1
    pi_x = optimize(env, -r1) # worst policy under R1

    # return values under R1
    J_1_pi_1 = policy_return(r1, pi_1, env)
    J_1_pi_x = policy_return(r1, pi_x, env)

    # canonicalizations and normalizations of R1
    can1 = canon_and_norm(r1, env)

    # interpolate between R1 and R2
    for r_i in interpolate(r1, r2, config['interpolation_steps']):
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
      # compute regrets (normalizing over max-min)
      interp_results['regret1'] = (J_1_pi_1 - J_1_pi_i) / (J_1_pi_1 - J_1_pi_x)
      interp_results['regret2'] = (J_i_pi_i - J_i_pi_1) / (J_i_pi_i - J_i_pi_y)

      # save results
      sample_results.append(interp_results)
    trail_results.append(sample_results)
  results.append(trail_results)
        
# save as JSON
with open(results_path, 'w') as f:
  json.dump({'config': config, 'results': results}, f)
