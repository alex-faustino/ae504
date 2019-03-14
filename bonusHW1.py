# Alex Faustino
# Bonus HW 1
# AE504 Optimal Aerospace Systems
# 14 March 2019

import numpy as np
#import matplotlib.pyplot as plt

from classes.roverGridworld import RoverGridWorldEnv

env = RoverGridWorldEnv()

reward_for_action = [env.step(state[0], action)[1] for action in env.action_space for state in env.state_space]
print (np.shape(reward_for_action))