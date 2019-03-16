# Alex Faustino
# Bonus HW 1
# AE504 Optimal Aerospace Systems
# 14 March 2019

import numpy as np

from classes.roverGridworld import RoverGridWorldEnv


heights = [.4, .6, .4, 4.9, .3, 1, 2.4,
           9, 7.2, .8, .4, 3, .4, 3.5,
           2.4, 5.9, 4.2, 7.9, 8.2, 2.4, 0,
           7.3, 3.6, 10.7, 9.8, 2.4, 3, 4.8,
           .5, .3, 5.4, 9.2, 2.4, 1.4, .3,
           5.5, 4.5, 3.9, 7.6, 5.2, 5.7, 1.3,
           0, 1.3, .2, .2, 2.1, 1, .6]

env = RoverGridWorldEnv(7, heights)

# Problem 1
# Find the reward for taking each action at every state
reward_for_action = [env.step(state[0], action)[1] for action in                                     env.action_space for state in env.state_space]

# Group by action
reward_for_action = np.reshape(reward_for_action, (5, -1))

# Problem 2
# part a
env.get_value(6,42)
print(env.value_funcs[(6,42)])
print(env.get_path(6,42))

# part b
env.get_value(10,42)
print(env.value_funcs[(10,42)])
print(env.get_path(10,42))

# part c
env.get_value(20,42)
print(env.value_funcs[(20,42)])
print(env.get_path(20,42))
