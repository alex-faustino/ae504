import math


class RoverGridWorldEnv():

    def __init__(self):
        # set action space
		# 0 = stay
		# 1 = North
		# 2 = East
		# 3 = South
        # 4 = West
        self.action_space = range(5)

        # creat flattened state space with heights
        heights = [0, 1.3, .2, .2, 2.1, 1, .6,
                   5.5, 4.5, 3.9, 7.6, 5.2, 5.7, 1.3,
                   .5, .3, 5.4, 9.2, 2.4, 1.4, .3,
                   7.3, 3.6, 10.7, 9.8, 2.4, 3, 4.8,
                   2.4, 5.9, 4.2, 7.9, 8.2, 2.4, 0,
                   9, 7.2, .8, .4, 3, .4, 3.5,
                   .4, .6, .4, 4.9, .3, 1, 2.4]

        self.state_space = list(zip(range(49), heights))

    def step(self, state, action):
        # define negative infinity
        neg_inf = float("-inf")

        # unflatten state
        a = action
        s = state
        x1 = s % 7
        x2 = s // 7

        # handle edge cases
        if (x1 == 0 and a == 4) or (x1 == 6 and a == 2) or (x2 == 0 and a == 1) or (x2 == 6 and a == 3):
            r = neg_inf
        # normal cases
        else:
            if a == 0:
                r = 0
                splus1 = s
            if a == 1:
                splus1 = s - 7
                r = self.get_reward(s, splus1)
            if a == 2:
                splus1 = s + 1
                r = self.get_reward(s, splus1)
            if a == 3:
                splus1 = s + 7
                r = self.get_reward(s, splus1)
            if a == 4:
                splus1 = s - 1
                r = self.get_reward(s, splus1)
            s = splus1

        return s, r
        
    def get_reward(self, s, splus1):
        # calculate reward (negative cost) of going from s to splus1
        return -1 - (self.state_space[splus1][1] - self.state_space[s][1])**2
