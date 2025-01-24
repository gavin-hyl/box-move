import numpy as np
from BoxEnv import Box, BoxMoveEnvironment

# -------------------------------------------------
# Environment and Initial State
# -------------------------------------------------
env = BoxMoveEnvironment(zone_sizes=[(2,3,4), (2,3,4)], horizon=3, gamma=1)

def create_random_initial_state(env, n_boxes, zone=0):
    """
    Randomly create a valid initial state (all boxes in zone 0).
    """
    boxes = []
    zone_size = env.zone_sizes[zone]

    tries = 0
    max_tries = 500  # Just to prevent infinite loops
    while len(boxes) < n_boxes and tries < max_tries:
        tries += 1

        # Random size (at least 1 cell in each dimension)
        # x_size = np.random.randint(1, zone_size[0] + 1)
        # y_size = np.random.randint(1, zone_size[1] + 1)
        # z_size = np.random.randint(1, zone_size[2] + 1)
        x_size = np.random.randint(1, 2)
        y_size = np.random.randint(1, 2)
        z_size = np.random.randint(1, 2)

        # Random position ensuring the box fits
        x_pos = np.random.randint(0, zone_size[0] - x_size + 1)
        y_pos = np.random.randint(0, zone_size[1] - y_size + 1)
        z_pos = np.random.randint(0, zone_size[2] - z_size + 1)

        candidate_box = Box.make(
            pos=np.array([x_pos, y_pos, z_pos]),
            size=np.array([x_size, y_size, z_size]),
            zone=zone
        )

        # Test if adding this new box is still a valid state
        test_boxes = boxes + [candidate_box]
        test_state = Box.state_from_boxes(test_boxes)
        if env._is_valid_state(test_state):
            boxes.append(candidate_box)

    # Create the final state from whatever valid boxes we have
    state = Box.state_from_boxes(boxes, t=0)
    return state


initial_state = create_random_initial_state(env, n_boxes=3, zone=0)

print("Initial State:")
print(Box.to_str(initial_state))

possible_actions = env.actions(initial_state)
print("Possible Actions:")
for act in possible_actions:
    print(f" - {act}")


# -------------------------------------------------
# NN Definition
# -------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.net(x)


# -------------------------------------------------
# Epsilon Greedy Policy
# -------------------------------------------------
import random
def select_action(dqn, state_vec, actions_list, epsilon=0.1):
    """
    - state_vec: 1D numpy array of shape (state_dim,)
    - actions_list: all possible actions from env.actions(state)
    - epsilon: exploration probability
    """
    if random.random() < epsilon:
        # Explore: pick random action
        return random.choice(actions_list)
    else:
        # Exploit: pick argmax Q
        state_t = torch.tensor(state_vec, dtype=torch.float).unsqueeze(0)  # shape (1, state_dim)
        # Evaluate Q-values for each possible action
        q_vals = dqn(state_t)  # shape (1, action_dim)
        _, best_action_idx = torch.max(q_vals, dim=1)
        return actions_list[best_action_idx.item()]


# -------------------------------------------------
