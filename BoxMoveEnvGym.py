import gym
import numpy as np
import matplotlib.pyplot as plt

from BoxMoveEnv import BoxMoveEnv
from BoxAction import BoxAction
from Constants import ZONE0, ZONE1

class BoxMoveEnvGym(gym.Env):
    """
    Gym wrapper for the BoxMoveEnv.
    
    The action space is defined as a Discrete space, where each integer corresponds to a
    specific (pos_from, pos_to) move from zone0 to zone1. The observation is taken from
    BoxMoveEnv.state_1d()
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, horizon=100, gamma=1, n_boxes=5):
        super().__init__()
        # Instantiate the underlying BoxMoveEnv
        self.env = BoxMoveEnv(horizon, gamma, n_boxes)

        # Define a discrete action space based on all possible moves from zone0 to zone1.
        total_actions = np.prod(ZONE0) * np.prod(ZONE1)
        self.action_space = gym.spaces.Discrete(int(total_actions))

        # Use the 1D state representation as the observation.
        initial_obs = self.env.state_1d()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=initial_obs.shape,
            dtype=initial_obs.dtype
        )

        # Build a mapping from discrete action index to a BoxAction.
        self._action_map = {}
        idx = 0
        for pos_from in np.ndindex(ZONE0):
            for pos_to in np.ndindex(ZONE1):
                self._action_map[idx] = BoxAction(pos_from, pos_to)
                idx += 1

    def reset(self, n_boxes=None):
        """
        Resets the environment.
        
        If n_boxes is provided, the environment is reset with that many boxes.
        Returns:
            obs (np.ndarray): The initial state observation.
        """
        if n_boxes is not None:
            self.env.reset(n_boxes)
        else:
            self.env.reset()
        return self.env.state_1d()

    def step(self, action):
        """
        Takes a discrete action and advances the environment.
        
        Args:
            action (int): The discrete action index.
        
        Returns:
            obs (np.ndarray): The next state observation.
            reward (float): The reward after taking the action.
            done (bool): True if the episode has ended.
            truncated (bool): True if the episode was truncated (not used here).
            info (dict): Additional information (e.g., if the chosen action was invalid).
        """
        # Map the discrete action index to a BoxAction.
        if action not in self._action_map:
            raise ValueError("Action index out of range.")
        chosen_action = self._action_map[action]

        # Check if the chosen action is among the valid actions.
        valid = any(
            chosen_action == va
            for va in self.env.actions()
        )
        if not valid:
            # Penalize for selecting an invalid action.
            penalty = -1
            obs = self.env.state_1d()
            done = False
            info = {"invalid_action": True}
            return obs, penalty, done, True, info

        # Perform the action in the underlying environment.
        reward = self.env.step(chosen_action)
        obs = self.env.state_1d()
        # Determine if the episode is done (i.e., reached the horizon or no valid actions remain).
        done = (self.env.state[-1] == self.env.horizon or len(self.env.actions()) == 0)
        info = {}
        return obs, reward, done, False, info

    def render(self, mode="human"):
        """
        Render the current state of the environment.
        """
        if mode == "human":
            self.env.visualize_scene()
        else:
            raise NotImplementedError(f"Render mode {mode} is not supported.")

    def close(self):
        """
        Closes the environment (e.g., closing matplotlib windows).
        """
        plt.close("all")
