import math


class RoverGridWorldEnv():

    def __init__(self, size, heights):
        # set action space
		# 0 = stay
		# 1 = North
		# 2 = East
		# 3 = South
        # 4 = West
        self.action_space = range(5)

        # creat flattened state space with heights
        self.size = size
        self.state_space = list(zip(range(size**2), heights))

        # initialize value function storage
        # define negative infinity
        self.neg_inf = float("-inf")
        self.value_funcs = {(0, s): self.neg_inf for s in range(49)}
        self.value_funcs[(0, 0)] = 0

        # initialize action list for each value function
        self.value_funcs_acts = {(0, s): [] for s in range(49)}
        self.value_funcs_acts[(0, 0)] = 0

    def step(self, state, action):

        # unflatten state
        a = action
        s = state
        size = self.size
        x1 = s % size
        x2 = s // size

        # handle edge cases
        if (x1 == 0 and a == 4) or (x1 == (size - 1) and a == 2) or (x2 == 0 and a == 3) or (x2 == (size - 1) and a == 1):
            r = self.neg_inf
            splus1 = s
        # normal cases
        else:
            if a == 0:
                r = 0
                splus1 = s
            if a == 1:
                splus1 = s + size
                r = self.get_reward(s, splus1)
            if a == 2:
                splus1 = s + 1
                r = self.get_reward(s, splus1)
            if a == 3:
                splus1 = s - size
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
        size = self.size
        x1 = s % size
        x2 = s // size

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
                temp_a = a

        self.value_funcs[(T,s)] = temp_v
        self.value_funcs_acts[(T,s)] = temp_a

        return temp_v

    def get_path(self, T, state):
        path = []
        s = state
        for t in range(T, 0, -1):
            a = self.value_funcs_acts[(t,s)]
            path.append(a)
            splus1, _ = self.step(s, a)
            s = splus1
            
        return path