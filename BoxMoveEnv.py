import numpy as np

from BoxAction import BoxAction
from Core import actions, transition

class BoxMoveEnv:
    def __init__(self, horizon=100, gamma=1):
        self.horizon = horizon
        self.gamma = gamma

    def actions(self, state: np.ndarray):
        return actions(state)
    
    def step(self, state: np.ndarray, action: BoxAction):
        state = transition(state, action)
        state[-1] += 1
    
    def reward(self):
        return self.occupancy()