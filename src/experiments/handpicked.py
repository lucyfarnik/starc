import itertools
import numpy as np
import json
from env.handpicked.epic_gridworlds import init_epic_gridworlds
from distance.canon import canon_and_norm
from distance.rollout import optimize, policy_return, policy_returns

def handpicked_experiment(results_path: str):
  # hyperparameters
  slippery = True

  dist_opts = [1, 2, float('inf')]

  np.random.seed(42)

  env, rewards = init_epic_gridworlds(slippery)

  # experiment starts here
  results = {}
  for (r1_name, r1), (r2_name, r2) in itertools.combinations(rewards.items(), 2):
    r_names = f'{r1_name}-{r2_name}'
    print(r_names)
    results[r_names] = {}

    # canonicalizations and normalizations of R1 and R2
    can1 = canon_and_norm(r1, env)
    can2 = canon_and_norm(r2, env)

    # compute the distances for all combinations of canon, norm, and dist
    for cn_name, r1_val in can1.items():
      r2_val = can2[cn_name]
      for d_ord in dist_opts:
        results[r_names][f'{cn_name}-{d_ord}'] = np.linalg.norm(
          (r1_val - r2_val).flatten(), d_ord
        )

    # calculate the best and worst policies under R1 and R2
    pi_1 = optimize(env, r1) # best policy under R1
    pi_x = optimize(env, -r1) # worst policy under R1
    pi_2 = optimize(env, r2) # best policy under R2
    pi_y = optimize(env, -r2) # worst policy under R2

    # compute return values
    J_1_pi_1, J_2_pi_1 = policy_returns([r1, r2], pi_1, env)
    J_1_pi_2, J_2_pi_2 = policy_returns([r1, r2], pi_2, env)
    J_1_pi_x = policy_return(r1, pi_x, env)
    J_2_pi_y = policy_return(r2, pi_y, env)
    # compute regrets (normalizing over max-min)
    results[r_names]['regret1'] = (J_1_pi_1 - J_1_pi_2) / (J_1_pi_1 - J_1_pi_x)
    results[r_names]['regret2'] = (J_2_pi_2 - J_2_pi_1) / (J_2_pi_2 - J_2_pi_y)

  # save as JSON
  with open(results_path, 'w') as f:
    json.dump(results, f)


if __name__ == '__main__':
  handpicked_experiment()
