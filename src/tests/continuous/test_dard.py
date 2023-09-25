import numpy as np
from continuous.canon.dard import dard_canon_cont
from tests.continuous.toy_env_cont import reward, env_info
from tests.continuous.consistency_test import consistency_test

def test_dard():
  canonicalized = dard_canon_cont(reward, env_info)

  # check that the canonicalized reward is consistent
  consistency_test(canonicalized, env_info)

  raise NotImplementedError
