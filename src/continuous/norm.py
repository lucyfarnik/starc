from _types import RewardCont, Space
from typing import Union
from utils import timed, sample_space
from multiprocessing import Process, Value, Lock
import ctypes

def run_rew_and_compare_w_max(reward: RewardCont,
                              max_sample: Value,
                              lock: Lock,
                              state_space: Space,
                              action_space: Space):
  """Executed by a process to sample a reward and compare it to the current maximum"""
  # sample s, a, s'
  s = sample_space(state_space)
  a = sample_space(action_space)
  s_prime = sample_space(state_space)

  sample_val = abs(reward(s, a, s_prime)) # |r(s,a,s')|
  # if is_weighted: sample_val *= env_info.trans_prob(s, a, s_prime) # weighted norms

  with lock:
    if sample_val > max_sample.value:
      max_sample.value = sample_val 

def find_max(reward: RewardCont,
             state_space: Space,
             action_space: Space,
             n_samples: int):
  """Finds the maximum of a continuous function (by sampling across many processes)"""
  # TODO try assuming the function is convex and differentiable in Torch; do optimization
  processes = []
  lock = Lock()
  max_sample = Value(ctypes.c_float, float('-inf'))
  run_func_args = (reward, max_sample, lock, state_space, action_space)

  for _ in range(n_samples):
    p = Process(target=run_rew_and_compare_w_max, args=run_func_args)
    p.start()
    processes.append(p)
  
  # wait for all processes to finish
  for p in processes: p.join()
  
  return max_sample

def run_rew_and_add_pow(reward: RewardCont,
                        sample_sum: Value,
                        lock: Lock,
                        state_space: Space,
                        action_space: Space,
                        ord: int):
  """Executed by a process to sample a reward and add |R|^p to the sum"""
  # sample s, a, s'
  s = sample_space(state_space)
  a = sample_space(action_space)
  s_prime = sample_space(state_space)

  sample_val = abs(reward(s, a, s_prime))**ord # |r(s,a,s')|^p
  # if is_weighted: sample_val *= env_info.trans_prob(s, a, s_prime) # weighted norms

  with lock:
    sample_sum.value += sample_val

@timed
def norm_cont(reward: RewardCont,
              state_space: Space,
              action_space: Space,
              ord: Union[int, float, str] = 2,
              n_samples: int = 10**3) -> float:
  """
    Takes the norm of a continuous function
    ord: 0, 1, 2, 'weighted_1', 'weighted_2', 'weighted_inf', 'inf'
  """

  if ord == 0: return 1 # baseline (no norm)

  # is_weighted = type(ord) is str and 'weighted' in ord
  # norm_ord = float(ord.split('_')[1]) if is_weighted else float(ord)

  # L_infty norm — find the maximum
  if ord == float('inf'):
    return find_max(reward, state_space, action_space, n_samples)

  processes = []
  lock = Lock()
  sample_sum = Value(ctypes.c_float, 0.0)
  run_func_args = (reward, sample_sum, lock, state_space, action_space, ord)

  for _ in range(n_samples):
    p = Process(target=run_rew_and_add_pow, args=run_func_args)
    p.start()
    processes.append(p)
  
  # volume of the domain, ie the integral of 1 over the domain
  state_space_volume = 1
  for interval in state_space:
    state_space_volume *= interval[1] - interval[0]
  action_space_volume = 1
  for interval in action_space:
    action_space_volume *= interval[1] - interval[0]
  domain_volume = (state_space_volume ** 2) * action_space_volume

  # wait for all processes to finish
  for p in processes: p.join()
  
  # (V / N * sum)^(1/p)
  # TODO check that including the volume in this way is the "normal" Lp norm
  # TODO (though it's definitely bilipschitz equivalent to it so doesn't really matter)
  return (domain_volume / n_samples * sample_sum)**(1/ord)
  