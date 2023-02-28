import numpy as np
from env import Env
from _types import Reward, Policy

# TODO fine tune to make it faster - is the max_iters right? maybe add a convergence return condition?
def optimize(
  env: Env,
  reward: Reward,
  max_iters=10000,
  epsilon=0.1,
  episode_len=100,
  learning_rate=1e-3,
) -> Policy:
  q_vals = np.zeros((env.n_s, env.n_a))

  for i in range(max_iters):
    # reset the episode every now and then so it doesn't get stuck
    if i % episode_len == 0:
      s = np.random.choice(env.states, p=env.init_dist)

    # behavior policy = epsilon-greedy
    if np.random.random() > epsilon:
      # a = np.random.choice(env.actions, p=softmax(q_vals[s]))
      a = q_vals[s].argmax()
    else:
      a = np.random.choice(env.actions)

    # sample next state
    s_next = np.random.choice(env.states, p=env.transition_dist[s, a])
    r = reward[s, a, s_next]

    # compute TD error and update Q value
    delta = r + env.discount * q_vals[s_next].max() - q_vals[s, a]
    q_vals[s, a] += learning_rate * delta

    # the next state becomes the current state
    s = s_next

  return q_vals.argmax(axis=-1)

# Monte Carlo estimation
# TODO this still has pretty high variance, having a static number of episodes and steps probably isn't ideal
def policy_returns(
  rewards: list[Reward],
  policy: Policy,
  env: Env,
  num_episodes=20,
  steps_per_episode=1000,
  compute_return_per_steps=10, # number of timesteps between return samples - see comment below
) -> list[float]:
  num_rs = len(rewards)

  # 2D array, first dim is different reward funcs, second dim is samples
  return_vals = [[] for _ in range(num_rs)]

  for _ in range(num_episodes):
    # init state
    s = np.random.choice(env.states, p=env.init_dist)
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
    
    # at the end we compute the return - we wanna make sure we take into account
    # what happened throughout the episode and not just at the start, so we
    # compute it starting at different start points

    # we're using steps_per_episode-100 as the end here so that the sum
    # doesn't get cut prematurely by the end of the episode which would
    # drag down the average
    #? a more accurate way to do this would be replace 100 with log_gamma(1e-4)
    for r_i, r_values in enumerate(episode_rewards): # for each return func
      for return_start in range(0, steps_per_episode-100, compute_return_per_steps):
        return_val = 0 # accumulator for return from return_start
        for i, r in enumerate(r_values[return_start:]):
          if i == 0: gamma_i = 1.0
          else: gamma_i *= env.discount
          
          if gamma_i < 1e-4: break # with discounts this heavy it's not worth computing
          return_val += gamma_i * r
        return_vals[r_i].append(return_val)

  return [sum(rs) / len(rs) for rs in return_vals]

# wrapper for the function above - takes just one reward function
def policy_return(reward: Reward, *args) -> Reward:
  return policy_returns([reward], *args)[0]
