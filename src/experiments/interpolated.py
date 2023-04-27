from multiprocessing import Pool
import os
import numpy as np
import json
from env import RandomEnv
from env.reward import random_reward, interpolate
from distance.canon import canon_and_norm
from distance.rollout import optimize, policy_return, policy_returns

config = {
  # hyperparams
  'num_envs': 16,
  'num_rewards': 16,
  'interpolation_steps': 16,
  'n_s': 32,
  'n_a': 4,
  'seed': 42,

  # env and reward settings
  'episodic': False,
  'discount': 0.95,
  'sparse': None,
  'state_dependent': False,
  'reward_only_in_terminal': False,
}
temp_dir = 'temp/interpolated'
dist_opts = [1, 2, float('inf')]

def interpolated_env_run(env_i):
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
        if cn_name not in can_i: continue
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
      regret1_norm = J_1_pi_1 - J_1_pi_x
      if regret1_norm == 0:
        if J_1_pi_1 - J_1_pi_i != 0:
          print("\n\nWarning: regret1_norm is 0, but regret1 is not 0\n\n")
        regret1_norm = 1
      regret2_norm = J_i_pi_i - J_i_pi_y
      if regret2_norm == 0:
        if J_i_pi_i - J_i_pi_1 != 0:
          print("\n\nWarning: regret2_norm is 0, but regret2 is not 0\n\n")
        regret2_norm = 1
      interp_results['regret1'] = (J_1_pi_1 - J_1_pi_i) / regret1_norm
      interp_results['regret2'] = (J_i_pi_i - J_i_pi_1) / regret2_norm

      # save results
      sample_results.append(interp_results)
    # save results for this reward pair into temp directory
    with open(f'{temp_dir}/env{env_i}_reward{reward_i}.json', 'w') as f:
      json.dump(sample_results, f)

def interpolated_experiment(results_path: str):
  np.random.seed(config['seed'])

  # create temp results directory
  os.makedirs(temp_dir, exist_ok=True)

  # experiment starts here
  with Pool() as p:
    p.map(interpolated_env_run, range(config['num_envs']))

  # collect results
  results = []
  for env_i in range(config['num_envs']):
    env_results = []
    for reward_i in range(config['num_rewards']):
      with open(f'{temp_dir}/env{env_i}_reward{reward_i}.json', 'r') as f:
        env_results.append(json.load(f))
    results.append(env_results)
          
  # save as JSON
  with open(results_path, 'w') as f:
    json.dump({'config': config, 'results': results}, f)
  
  # remove temp results directory
  os.system(f'rm -rf {temp_dir}')

if __name__ == '__main__':
  interpolated_experiment()
