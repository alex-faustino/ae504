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
        heights = [.4, .6, .4, 4.9, .3, 1, 2.4,
                   9, 7.2, .8, .4, 3, .4, 3.5,
                   2.4, 5.9, 4.2, 7.9, 8.2, 2.4, 0,
                   7.3, 3.6, 10.7, 9.8, 2.4, 3, 4.8,
                   .5, .3, 5.4, 9.2, 2.4, 1.4, .3,
                   5.5, 4.5, 3.9, 7.6, 5.2, 5.7, 1.3,
                   0, 1.3, .2, .2, 2.1, 1, .6]

        self.state_space = list(zip(range(49), heights))

        # initialize value function storage
        # define negative infinity
        self.neg_inf = float("-inf")
        self.value_funcs = {(0, s): self.neg_inf for s in range(49)}
        self.value_funcs[(0, 0)] = 0

    def step(self, state, action):

        # unflatten state
        a = action
        s = state
        x1 = s % 7
        x2 = s // 7

        # handle edge cases
        if (x1 == 0 and a == 4) or (x1 == 6 and a == 2) or (x2 == 0 and a == 3) or (x2 == 6 and a == 1):
            r = self.neg_inf
            splus1 = s
        # normal cases
        else:
            if a == 0:
                r = 0
                splus1 = s
            if a == 1:
                splus1 = s + 7
                r = self.get_reward(s, splus1)
            if a == 2:
                splus1 = s + 1
                r = self.get_reward(s, splus1)
            if a == 3:
                splus1 = s - 7
                r = self.get_reward(s, splus1)
            if a == 4:
                splus1 = s - 1
                r = self.get_reward(s, splus1)

        return splus1, r
        
    def get_reward(self, s, splus1):
        # calculate reward (negative cost) of going from s to splus1
        return -1 - (self.state_space[splus1][1] - self.state_space[s][1])**2

    def get_value(self, T, state):
        # unflatten state
        s = state
        x1 = s % 7
        x2 = s // 7

        # check if value function has already been solved for
        if (T,s) in self.value_funcs.keys():
            return self.value_funcs[(T,s)]
        
        # check if terminal state can't be reached in T steps
        if (x1 + x2) > T:
            self.value_funcs[(T,s)] = self.neg_inf
            return self.neg_inf

        # recursively find value function
        temp_v = self.neg_inf
        for a in self.action_space:
            splus1, r = self.step(s, a)
            v = r + self.get_value(T - 1, splus1)
            if v > temp_v:
                temp_v = v

        self.value_funcs[(T,s)] = temp_v
        return temp_v