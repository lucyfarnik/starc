# Continuous experiment
from multiprocessing import Pool
import numpy as np
from continuous.distance import canon_and_norm_cont
from continuous.norm import norm_cont
from continuous.env import ReacherEnv
from continuous.rewards import (
    GroundTruthReward, NegativeGroundReward, PotentialShapedReward,
    RandomReward, SPrimeReward, SecondPeakReward,
    SemanticallyIdenticalReward
)

config = {
  'seed': 42,
  'discount': 0.99,
  'temp_dir': 'temp/continuous',
}

def continuous_experiment(results_path: str):
  np.random.seed(config['seed'])

  # create the rewards
  ground_truth_env = ReacherEnv(GroundTruthReward(), config['discount'])
  non_ground_envs = {
    'potential_shaped': ReacherEnv(PotentialShapedReward(), config['discount']),
    's_prime': ReacherEnv(SPrimeReward(), config['discount']),
    'second_peak': ReacherEnv(SecondPeakReward(), config['discount']),
    'semantically_identical': ReacherEnv(SemanticallyIdenticalReward(), config['discount']),
    'negative_ground': ReacherEnv(NegativeGroundReward(), config['discount']),
    'random': ReacherEnv(RandomReward(), config['discount']),
  }

  results: dict[str, float] = {}

  # loop over all rewards and compute their distance from the ground truth
  standardized_gt = canon_and_norm_cont(ground_truth_env.reward_func_curried,
                                        ground_truth_env.env_info)
  for r2_name, e2 in non_ground_envs.items(): # TODO parallelize as much as possible
    standardized_r2 = canon_and_norm_cont(e2.reward_func_curried, e2.env_info)

    for cn_name in standardized_gt:
      cn_gt = standardized_gt[cn_name]
      cn_r2 = standardized_r2[cn_name]

      for d_ord in [1, 2, float('inf')]:
        diff_func = lambda *args: cn_gt(*args) - cn_r2(*args)

        dist = norm_cont(diff_func, ReacherEnv.state_space, ReacherEnv.act_space, d_ord)

        results_key = f'{r2_name}-{cn_name}-{d_ord}'
        print(f'{results_key}: {dist}')
        results[results_key] = dist
        with open(f"{config['temp_dir']}/{results_key}.txt", 'w') as f:
          f.write(str(dist))

if __name__ == '__main__':
  continuous_experiment()
