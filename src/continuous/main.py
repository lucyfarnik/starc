# Continuous experiment
from multiprocessing import Pool
import numpy as np
from continuous.distance import canon_and_norm_cont
from continuous.norm import norm_cont
from continuous.env import get_vec_env, train_agent, predict_next_state
from _types import EnvInfoCont, RewardCont

config = {
  'seed': 42,
  'discount': 0.99, #! MAKE SURE THIS GETS USED EVERYWHERE
  'temp_dir': 'temp/continuous',
}

def continuous_experiment(results_path: str):
  np.random.seed(config['seed'])

  # create the environment
  env_vec = get_vec_env()
  env_info = EnvInfoCont(
    trans_dist=lambda s, a: predict_next_state(env_vec.envs[0], s, a),
    discount=config['discount'],
    state_space=env_vec.observation_space, # TODO convert me
    action_space=env_vec.action_space, # TODO convert me
    state_vals=None, # TODO
  )

  # create the rewards
  # TODO all of this
  ground_truth_reward = None
  potential_shaped_reward = None
  s_redistributed_reward = None
  second_peak_reward = None
  same_seman_diff_struct_reward = None
  negative_ground_reward = None
  random_reward = None

  non_ground_rewards = {
    'potential_shaped': potential_shaped_reward,
    's_redistributed': s_redistributed_reward,
    'second_peak': second_peak_reward,
    'same_seman_diff_struct': same_seman_diff_struct_reward,
    'negative_ground': negative_ground_reward,
    'random': random_reward,
  }

  results: dict[str, float] = {}

  # loop over all rewards and compute their distance from the ground truth
  standardized_gt = canon_and_norm_cont(ground_truth_reward, env_info)
  for r2_name, r2 in non_ground_rewards.items(): #! TODO parallelize as much as possible
    standardized_r2 = canon_and_norm_cont(r2, env_info)

    for cn_name in standardized_gt:
      cn_gt = standardized_gt[cn_name]
      cn_r2 = standardized_r2[cn_name]

      for d_ord in [1, 2, float('inf')]:
        diff_func = lambda s, a, sp: cn_gt(s, a, sp) - cn_r2(s, a, sp)

        dist = norm_cont(diff_func, env_info, d_ord)

        results_key = f'{r2_name}-{cn_name}-{d_ord}'
        print(f'{results_key}: {dist}')
        results[results_key] = dist
        with open(f"{config['temp_dir']}/{results_key}.txt", 'w') as f:
          f.write(str(dist))

if __name__ == '__main__':
  continuous_experiment()