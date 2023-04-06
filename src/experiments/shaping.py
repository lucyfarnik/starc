# testing out the files in shaping.py - is "shaped" EPIC better than regular EPIC?
import numpy as np
import json
from env import RandomEnv
from env.reward import random_reward, interpolate
from distance.canon import canon_and_norm
from distance.rollout import optimize, policy_return, policy_returns
from env import RandomEnv
from distance.distance_funcs import epic, epic_torch
from distance.shaping import shaped_distance, maximal_transition_dist

def shaping_experiment(results_path: str):
  # hyperparams
  num_envs = 4
  num_rewards = 4
  interpolation_steps = 4
  n_s = 64
  n_a = 8
  sparse_rs = False

  # experiment starts here
  results = []
  for env_i in range(num_envs): # each trial comes with its own environment
    env = RandomEnv(n_s, n_a=8)

    for reward_i in range(num_rewards): # sampling different rewards
      print(f"Environment {env_i}, rewards {reward_i}")
      sample_results = []

      # generate random rewards
      r1, r2 = random_reward(env, sparse_rs), random_reward(env, sparse_rs)

      # optimal policies for R1
      pi_1 = optimize(env, r1) # best policy under R1
      pi_x = optimize(env, -r1) # worst policy under R1

      # return values under R1
      J_1_pi_1 = policy_return(r1, pi_1, env)
      J_1_pi_x = policy_return(r1, pi_x, env)

      for r_i in interpolate(r1, r2, interpolation_steps):
        interp_results = {}

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

        # compute distances
        norm1 = r1 / np.linalg.norm(r1.flatten(), 2)
        norm_i = r_i / np.linalg.norm(r_i.flatten(), 2)
        interp_results['None'] = np.linalg.norm((norm1 - norm_i).flatten(), 2)
        interp_results['EPIC'] = epic(r1, r_i, env)
        interp_results['Shaped EPIC'] = shaped_distance(r1, r_i, env, epic)
        interp_results['Maximal Shaped EPIC'] = maximal_transition_dist(r1, r_i, env, epic_torch)

        sample_results.append(interp_results)
      results.append(sample_results)
          
  # save as JSON
  with open(results_path, 'w') as f:
    json.dump(results, f)

if __name__ == '__main__':
  shaping_experiment()
  