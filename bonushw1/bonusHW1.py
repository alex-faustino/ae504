# Alex Faustino
# Bonus HW 1
# AE504 Optimal Aerospace Systems
# 14 March 2019

import numpy as np
import matplotlib.pyplot as plt

from classes.roverGridworld import RoverGridWorldEnv

env = RoverGridWorldEnv()

# Problem 1
# Find the reward for taking each action at every state
reward_for_action = [env.step(state[0], action)[1] for action in                                     env.action_space for state in env.state_space]

# Group by action
reward_for_action = np.reshape(reward_for_action, (5, -1))

# Problem 2
# part a
env.get_value(6,42)
print(env.value_funcs[(6,42)])

# part b
env.get_value(10,42)
print(env.value_funcs[(10,42)])

# part c
env.get_value(20,42)
print(env.value_funcs[(20,42)])