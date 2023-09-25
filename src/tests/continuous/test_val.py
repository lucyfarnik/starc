import numpy as np
from continuous.canon.val import val_canon_cont
from tests.continuous.toy_env_cont import reward, env_info
from tests.continuous.consistency_test import consistency_test

def test_val():
  canonicalized = val_canon_cont(reward, env_info)

  # check that the canonicalized reward is consistent
  consistency_test(canonicalized, env_info)

  # make sure it implements the correct formula (E[R(s,a,S') - V(s) + gamma*V(S')])
  raise NotImplementedError
