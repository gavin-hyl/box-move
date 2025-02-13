import Box
from Dense import dense_state, dense_action
from Constants import GEO_DIM, BOX_DIM, REMOVE_DIR, ZONE_SIZES, ZONE0, ZONE1
import Core

import numpy as np

import gym
from gym import spaces


class BoxMoveEnvGym(gym.Env):
    """
    A Gym-compatible wrapper for BoxMoveEnvironment.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, horizon=100, gamma=1, n_boxes=5):
        """
        Args:
            horizon (int): maximum number of steps in an episode
            gamma (float): discount factor
        """
        super().__init__()
        self.horizon = horizon
        max_boxes = np.prod(ZONE0)
        max_possible_actions = max_boxes * np.prod(ZONE1)
        flat_obs_dim = BOX_DIM * max_boxes + 1  # +1 for time

        # Required by Gym
        high = np.full((flat_obs_dim,), fill_value=max_boxes, dtype=int)
        self.observation_space = spaces.Box(
            low=-high, high=high, shape=(flat_obs_dim,), dtype=int
        )
        self.action_space = spaces.Discrete(max_possible_actions)
        self.gamma = gamma
        self.state = Core.random_initial_state(n_boxes)
        self._n_boxes = n_boxes

    def reset(self, seed=None, options=None):
        """
        Resets the environment to a start state.
        """
        super().reset(seed=seed)
        self.state = Core.pad_boxes(Core.random_initial_state(self._n_boxes))
        return self.state, None

    def step(self, action_idx: int):
        """
        Applies the chosen action and steps the environment forward.
        Gym step() returns (obs, reward, terminated, truncated, info).
        """
        valid_actions = Core.actions(self.state)
        all_actions = Core.all_actions()
        terminated, truncated = False, False
        if not valid_actions:
            reward = 0
            terminated = True
            info = {"reason": "no_valid_moves"}
        else:
            truncated = self.horizon == self.state[-1]
            if action_idx >= len(all_actions) or (chosen_action := all_actions[action_idx]) not in valid_actions:
                terminated = True
                reward = -1.0
                info = {"reason": "invalid_action_index"}
            else:
                self.state = Core.transition(self.state, chosen_action, step_time=True)
                reward = Core.occupancy(self.state)  # placeholder reward
                info = {"chosen_action": str(chosen_action)}
        obs = self.state

        return obs, reward, terminated, truncated, info


    def render(self, mode="human"):
        """
        Optional: Provide any rendering of the environment state if you want.
        """
        print("State")
        print(dense_state(self.state)[0])
        print(dense_state(self.state)[1])
        print("Actions")
        for rep in dense_action(self.state, self._valid_actions[0]):
            print(rep)

    def close(self):
        """
        Clean up if needed.
        """
        pass