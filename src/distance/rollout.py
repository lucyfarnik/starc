import numpy as np
from env import Env
from _types import Reward, Policy

# Q learning - not fine tuned, may have too many or too few iter steps
# def optimize(
#   env: Env,
#   reward: Reward,
#   max_iters=50000, # TODO try increasing
#   epsilon=0.1,
#   episode_len=100,
#   learning_rate=1e-3,
# ) -> Policy:
#   q_vals = np.zeros((env.n_s, env.n_a))

#   for i in range(max_iters):
#     # reset the episode every now and then so it doesn't get stuck
#     if i % episode_len == 0:
#       s = np.random.choice(env.states, p=env.init_dist)

#     # behavior policy = epsilon-greedy
#     if np.random.random() > epsilon:
#       # a = np.random.choice(env.actions, p=softmax(q_vals[s]))
#       a = q_vals[s].argmax()
#     else:
#       a = np.random.choice(env.actions)

#     # sample next state
#     s_next = np.random.choice(env.states, p=env.transition_dist[s, a])
#     r = reward[s, a, s_next]

#     # compute TD error and update Q value
#     delta = r + env.discount * q_vals[s_next].max() - q_vals[s, a]
#     q_vals[s, a] += learning_rate * delta

#     # the next state becomes the current state
#     s = s_next

#   return q_vals.argmax(axis=-1)

# value iteration
def optimize(env: Env, reward: Reward, convergence_thresh=1e-5) -> Policy:
  state_vals = np.zeros(env.n_s)

  for _ in range(10000):
    cond_p = env.transition_dist * (reward + env.discount * state_vals[None, None, :])
    new_vals = cond_p.sum(axis=2).max(axis=1)
    diff = state_vals - new_vals
    state_vals = new_vals
    if np.linalg.norm(diff, 2) < convergence_thresh: break
  
  return cond_p.sum(axis=2).argmax(axis=1)

# Monte Carlo estimation
def policy_returns(
  rewards: list[Reward],
  policy: Policy,
  env: Env,
  discount_thresh=1e-5,
) -> list[float]:
  # beyond this point, the discounts get so heavy that it's not worth computing
  steps_per_episode = round(np.log(discount_thresh) / np.log(env.discount))

  num_rs = len(rewards)

  # 2D array, first dim is different reward funcs, second dim is samples
  return_vals = [[] for _ in range(num_rs)]

  for _ in range(env.n_a): # do multiple repetitions to ensure low variance
    for episode_i in range(env.n_s):
      # init state - we want to have one episode for each possible starting state
      s = episode_i
      episode_rewards = [[] for _ in range(num_rs)] # same dims as return_vals

      for _ in range(steps_per_episode):
        # # sample action from policy
        # a = np.random.choice(env.actions, p=policy[s])
        a = policy[s]

        # next state
        s_next = np.random.choice(env.states, p=env.transition_dist[s, a])
        for i, r in enumerate(rewards):
          episode_rewards[i].append(r[s, a, s_next])
        s = s_next
      
      # at the end we compute the discounted return return
      for r_i, r_values in enumerate(episode_rewards): # for each return func
        return_val = 0 # accumulator for the return
        for i, r in enumerate(r_values):
          if i == 0: gamma_i = 1.0
          else: gamma_i *= env.discount
          return_val += gamma_i * r
        return_vals[r_i].append(return_val)

  return [sum(rs) / len(rs) for rs in return_vals]
  
# wrapper for the function above - takes just one reward function
def policy_return(reward: Reward, *args, **kwargs) -> Reward:
  return policy_returns([reward], *args, **kwargs)[0]
