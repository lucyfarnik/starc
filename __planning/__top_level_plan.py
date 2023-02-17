# from _ in range(number_of_trials):
#   draw environment from distribution_of_environments
#     for _ in range(number_of_samples):
#       generate two orthogonal reward functions R1, R2    # rewards with iid Gaussian entries are very likely to be orthogonal
#       let Ri be interpolated from R2 to R1:
#       for d in list_of_reward_distance_metrics:
#         save d(R1,Ri)
#       find the optimal R1 policy pi_1, and the optimal Ri policy pi_i
#       find worst possible R1 policy pi_x, and the worst possible Ri policy pi_y
#       save (J_1(pi_1) - J_1(pi_i))/(J_1(pi_1) - J_1(pi_x))  # the divisions etc normalise the regret to lie between 0 and 1
#       save (J_i(pi_i) - J_i(pi_1))/(J_i(pi_i) - J_i(pi_y))  # ditto

for _ in range(1000):
  env = random_env() #done
  for _ in range(1000):
    r1, r2 = random_R(env), random_R(env) #done
    pi_1 = optimize(env, r1) #done
    pi_x = optimize(env, -r1) #done
    for r_i in interpolated(r1, r2): #done
      for d in distances: #done
        save(d(r_1, r_i, env)) #TODO
      
      pi_i = optimize(env, r_i)
      pi_y = optimize(env, -r_i)
      J_1_pi_1 = policy_return(r1, pi_1, env) #TODO
      # ...
      save(regrets) #TODO

