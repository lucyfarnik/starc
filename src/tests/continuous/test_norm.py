import numpy as np
from continuous.norm import norm_cont
from continuous.canon.val import val_canon_cont
from tests.continuous.toy_env_cont import reward, env_info

def test_norm():
  canonicalized = val_canon_cont(reward, env_info)

  # check that the norm of the function is consistent
  l2_norm = norm_cont(canonicalized, env_info)
  for _ in range(3):
    val2 = norm_cont(canonicalized, env_info)
    assert abs(l2_norm - val2) < 1e-2 * l2_norm

  l_inf_norm = norm_cont(canonicalized, env_info, float('inf'))
  for _ in range(3):
    val2 = norm_cont(canonicalized, env_info, float('inf'))
    assert abs(l_inf_norm - val2) < 1e-2 * l_inf_norm

  l1_weighted_norm = norm_cont(canonicalized, env_info, 'weighted_1')
  for _ in range(3):
    val2 = norm_cont(canonicalized, env_info, 'weighted_1')
    assert abs(l1_weighted_norm - val2) < 1e-2 * l1_weighted_norm

  # check that the values actually make sense
  raise NotImplementedError
