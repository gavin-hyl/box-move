from Core import BoxMoveEnvironment, Box, STATE_DIM

import numpy as np

import gym
from gym import spaces
from copy import deepcopy


class BoxMoveEnvGym(gym.Env):
    """
    A Gym-compatible wrapper for BoxMoveEnvironment.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, zone_sizes, horizon=100, gamma=1, max_boxes=10):
        """
        Args:
            zone_sizes (list of np.array): dimension of each zone, e.g. [np.array([5,5,5]), np.array([5,5,5])]
            horizon (int): maximum number of steps in an episode
            gamma (float): discount factor (not directly used here for environment but relevant for RL)
            max_boxes (int): maximum number of boxes your environment can have
        """
        super().__init__()
        
        # Create your internal environment
        self.internal_env = BoxMoveEnvironment(horizon, gamma)
        self._max_boxes = zone_sizes[0][0] * zone_sizes[0][1] * zone_sizes[0][2]
        self.horizon = horizon
        self.max_possible_actions = 200
        self.dim_per_box = STATE_DIM  # 7 in your code
        self.flat_obs_dim = self.dim_per_box * max_boxes + 1  # +1 for time

        # We'll build the final observation_space as a Box space:
        # - A naive approach sets some numeric bounds. 
        high = np.full((self.flat_obs_dim,), fill_value=999999, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high, high=high, shape=(self.flat_obs_dim,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(self.max_possible_actions)

        self._valid_actions = []

        self.state = None

    def _flatten_state(self, state: np.array) -> np.array:
        """
        Given the environment's `state` (which is [box_1, box_2, ..., box_n, time]),
        convert it into a fixed-size 1D float32 array.
        If there are fewer than max_boxes, we pad with null boxes. 
        """
        boxes = Box.boxes_from_state(state)
        t = state[-1]

        # Build a new array that can hold up to max_boxes boxes:
        flattened = np.zeros(self.flat_obs_dim, dtype=np.float32)

        # Each box has self.dim_per_box entries, plus the last entry is time.
        # Fill in the actual boxes, up to `max_boxes`.
        for i, b in enumerate(boxes):
            if i >= self._max_boxes:
                # if we exceed max_boxes, break or handle it some way
                break
            start_idx = i * self.dim_per_box
            end_idx = start_idx + self.dim_per_box
            flattened[start_idx:end_idx] = b  # b is length self.dim_per_box

        # Place time as the last entry
        flattened[-1] = float(t)
        return flattened

    def _compute_valid_actions(self, state: np.array):
        """
        Compute the valid actions from the underlying environment's `actions(...)`.
        Returns a list of MoveBox objects.
        """
        return self.internal_env.actions(state)

    def reset(self, seed=None, options=None):
        """
        Resets the environment to a start state.
        """
        super().reset(seed=seed)

        box0 = Box.make(np.array([0,0,0]), np.array([1,1,1]), 0)
        init_state = Box.state_from_boxes([box0], t=0)

        self.state = init_state
        # Return the flattened observation
        return self._flatten_state(self.state), None

    def step(self, action_idx: int):
        """
        Applies the chosen action and steps the environment forward.
        Gym step() returns (obs, reward, done, truncated, info).
        """
        # 1. Map action_idx (0..max_possible_actions-1) to a valid MoveBox action.
        #    If action_idx >= len(_valid_actions), we treat it as "do nothing" or an invalid action.
        valid_actions = self._compute_valid_actions(self.state)
        self._valid_actions = valid_actions  # store them for debugging / reference

        if len(valid_actions) == 0:
            # No valid moves, so let's define a 'null' action => environment does not change
            next_state = deepcopy(self.state)
            reward = 0
            done = True
            info = {"reason": "no_valid_moves"}
        else:
            if action_idx >= len(valid_actions):
                # Chosen an invalid action index => can penalize or do nothing
                chosen_action = None
                # We'll just skip and do "no-op"
                next_state = deepcopy(self.state)
                reward = -1.0
                done = False
                info = {"reason": "invalid_action_index"}
            else:
                chosen_action = valid_actions[action_idx]
                # 2. Step the environment using your internal_env.step(...)
                next_state, step_reward, done = self.internal_env.step(self.state, chosen_action)

                # If you'd like to shape rewards or customize them:
                # e.g. we can track how much occupancy changed in zone=1, etc.
                reward = step_reward  # your environment returns 0 or occupancy if done, etc.
                info = {"chosen_action": str(chosen_action)}

        # 3. Update self.state
        self.state = next_state

        # 4. Build the next observation
        obs = self._flatten_state(self.state)

        return obs, reward, done, (self.horizon == self.state[-1]), info

    def render(self, mode="human"):
        """
        Optional: Provide any rendering of the environment state if you want.
        """
        print("State:", Box.to_str(self.state))
        if len(self._valid_actions) > 0:
            for i, action in enumerate(self._valid_actions):
                print(f"{i}. {action}")
        else:
            print("No valid actions")

    def close(self):
        """
        Clean up if needed.
        """
        pass