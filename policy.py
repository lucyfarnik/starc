import numpy as np
from jax import jit
import jax.numpy as jnp
from jax import random
from env import Env
from _types import Reward, Policy

# TODO fine tune to make it faster - is the max_iters right? maybe add a convergence return condition?
def optimize(
  env: Env, reward: Reward, rand_key: random.KeyArray,
  max_iters=10000,
  epsilon=0.1,
  episode_len=100,
  learning_rate=1e-3,
) -> tuple[Policy, random.KeyArray]:
  q_vals = np.zeros((env.n_s, env.n_a))

  for i in range(max_iters):
    # reset the episode every now and then so it doesn't get stuck
    if i % episode_len == 0:
      rand_key, subkey = random.split(rand_key)
      s = random.choice(subkey, env.states, p=env.init_dist)

    # behavior policy = epsilon-greedy
    rand_key, subkey = random.split(rand_key)
    if random.uniform(subkey) > epsilon:
      # a = np.random.choice(env.actions, p=softmax(q_vals[s]))
      a = q_vals[s].argmax()
    else:
      rand_key, subkey = random.split(rand_key)
      a = random.choice(subkey, env.actions)

    # sample next state
    rand_key, subkey = random.split(rand_key)
    s_next = random.choice(subkey, env.states, p=env.transition_dist[s, a])
    r = reward[s, a, s_next]

    # compute TD error and update Q value
    delta = r + env.discount * q_vals[s_next].max() - q_vals[s, a]
    q_vals[s, a] += learning_rate * delta

    # the next state becomes the current state
    s = s_next

  return jnp.array(q_vals.argmax(axis=-1))
optimize = jit(optimize)

# Monte Carlo estimation
# TODO this still has pretty high variance, having a static number of episodes and steps probably isn't ideal
# TODO add option to pass in multiple rewards
def policy_return(
  reward: Reward, policy: Policy, env: Env, rand_key: random.KeyArray,
  num_episodes=10,
  steps_per_episode=1000,
  compute_return_per_steps=10, # number of timesteps between return samples - see comment below
) -> tuple[float, random.KeyArray]:
  return_vals = []

  for _ in range(num_episodes):
    # init state
    rand_key, subkey = random.split(rand_key)
    s = random.choice(subkey, env.states, p=env.init_dist)
    episode_rewards = []

    for _ in range(steps_per_episode):
      # # sample action from policy
      # a = np.random.choice(env.actions, p=policy[s])
      a = policy[s]

      # next state
      rand_key, subkey = random.split(rand_key)
      s_next = random.choice(subkey, env.states, p=env.transition_dist[s, a])
      episode_rewards.append(reward[s, a, s_next])
      s = s_next
    
    # at the end we compute the return - we wanna make sure we take into account
    # what happened throughout the episode and not just at the start, so we
    # compute it starting at different start points

    # we're using steps_per_episode-100 as the end here so that the sum
    # doesn't get cut prematurely by the end of the episode which would
    # drag down the average
    #? a more accurate way to do this would be replace 100 with log_gamma(1e-4)
    for return_start in range(0, steps_per_episode-100, compute_return_per_steps):
      return_val = 0 # accumulator for return from return_start
      for i, r in enumerate(episode_rewards[return_start:]):
        if i == 0: gamma_i = 1.0
        else: gamma_i *= env.discount
        
        if gamma_i < 1e-4: break # with discounts this heavy it's not worth computing
        return_val += gamma_i * r
      return_vals.append(return_val)

  return sum(return_vals) / len(return_vals)
policy_return = jit(policy_return)
