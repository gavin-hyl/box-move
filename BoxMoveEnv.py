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
        """
        Takes a step in the simulator.

        Args:
            state (np.ndarray): the current state of the environment.
            action (BoxAction): the action to take.
        
        Returns:
            tuple: a tuple containing the next state of the environment, the
            reward received from taking the action, and a boolean indicating
            whether the episode has ended.
        """
        state = transition(state, action)
        state[-1] += 1
        return state, self.reward(), (state[-1] == self.horizon)
    
    def reward(self):
        return self.occupancy()