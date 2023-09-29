from _types import RewardCont, TransCont, Space
from typing import Union
from utils import timed, sample_space
from joblib import Parallel, delayed
from joblib.externals.loky.process_executor import BrokenProcessPool

def sample_reward_worker(reward: RewardCont,
                         trans_dist: TransCont,
                         state_space: Space,
                         action_space: Space):
  s = sample_space(state_space)
  a = sample_space(action_space)
  s_prime = trans_dist(s, a) #!!!!!!
  return reward(s, a, s_prime)

@timed
def norm_cont(reward: RewardCont,
              trans_dist: TransCont,
              state_space: Space,
              action_space: Space,
              ord: Union[int, float, str] = 2,
              n_samples: int = 10**3) -> float:
  if ord == 0: 
    return 1

  try:
    results = Parallel(n_jobs=-1, backend='threading')(
      delayed(sample_reward_worker)(reward, trans_dist, state_space, action_space)
      for _ in range(n_samples))
  except BrokenProcessPool:
    print("Failed to parallelize due to serialization issues.")
    return float('-inf')
  
  # take the absolute value of each item
  results = [abs(result) for result in results]
  # if is_weighted: sample_val *= env_info.trans_prob(s, a, s_prime) # weighted norms

  if ord == float('inf'):
    return max(results) 

  sample_sum = sum([r**ord for r in results])

  # state_space_volume = 1
  # for interval in state_space:
  #   if interval[0] == interval[1]: # prevents division by zero later on
  #     continue
  #   state_space_volume *= interval[1] - interval[0]
  # action_space_volume = 1
  # for interval in action_space:
  #   if interval[0] == interval[1]: # prevents division by zero later on
  #     continue
  #   action_space_volume *= interval[1] - interval[0]
  # domain_volume = (state_space_volume ** 2) * action_space_volume

  # domain_volume = 1 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  # return (domain_volume / n_samples * sample_sum)**(1/ord)
  return (sample_sum / n_samples)**(1/ord)
