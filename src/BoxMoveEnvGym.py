import copy
import random
import numpy as np
import gym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Box import Box
from src.BoxAction import BoxAction
from src.Constants import ZONE0, ZONE1, zone0_dense_cpy, zone1_dense_cpy, BOX_DIM

class BoxMoveEnvGym(gym.Env):
    """
    A gym environment for box movement tasks with 3D state representation.
    """
    metadata = {"render.modes": ["human"]}

    # ==========================================================================
    #                               Initialization
    # ==========================================================================
    def __init__(self, horizon=100, gamma=1, n_boxes=5, seed=None):
        super().__init__()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.horizon = horizon
        self.gamma = gamma
        self.t = 0  # time step
        self.boxes = []  # now a list of Box objects
        self.zone0_top = set()
        self.zone1_top = set()
        self.valid_actions = []
        self.n_boxes = n_boxes
        self._action_1d_map = {}
        
        # Define a discrete action space based on all possible moves from zone0 to zone1
        total_actions = np.prod(ZONE0) * np.prod(ZONE1)
        self.action_space = gym.spaces.Discrete(int(total_actions))
        
        # Determine the maximum number of boxes for state representation
        self._max_boxes = np.prod(ZONE0)
        # Each box is represented by 8 integers: pos (3), size (3), zone (1), val (1)
        self._box_dim = BOX_DIM

        # Build a mapping from discrete action index to a BoxAction
        self._action_map = {}
        idx = 0
        for pos_from in np.ndindex(ZONE0):
            for pos_to in np.ndindex(ZONE1):
                self._action_map[idx] = BoxAction(pos_from, pos_to)
                idx += 1

        # Define the observation space as a Dict with two keys
        self.observation_space = gym.spaces.Dict({
            "state_zone0": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, *ZONE0), dtype=np.float32),
            "state_zone1": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, *ZONE1), dtype=np.float32)
        })
        
        # Initialize the environment
        self.reset(n_boxes=n_boxes, seed=seed)

    # ==========================================================================
    #                            Resetting Functions
    # ==========================================================================
    def reset(self, n_boxes=None, seed=None):
        """
        Resets the environment.
        
        If n_boxes is provided, the environment is reset with that many boxes.
        Returns:
            obs (dict): A dictionary with keys "state_zone0" and "state_zone1" containing
                        the 3D state representations (with an added channel dimension).
        """
        if n_boxes is None:
            n_boxes = self.n_boxes

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.boxes = []
        tries = 0
        max_tries = 1000
        self.zone0_top = set()
        self.zone1_top = set()

        # Initialize the available "top" positions for each zone.
        for i, zone_size in enumerate([ZONE0, ZONE1]):
            for x in range(zone_size[0]):
                for y in range(zone_size[1]):
                    if i == 0:
                        self.zone0_top.add((x, y, 0))
                    else:
                        self.zone1_top.add((x, y, 0))

        # Try to place boxes in zone 0.
        while len(self.boxes) < n_boxes and tries < max_tries:
            tries += 1
            pos_choice = random.choice(list(self.zone0_top))
            # Do not allow placement if the z coordinate would exceed the zone's height.
            if pos_choice[2] >= ZONE0[2]:
                continue
            # Choose a random size that fits inside zone0 from the chosen position.
            size = np.random.randint(1, np.array(ZONE0) - np.array(pos_choice) + 1, size=3)
            candidate_box = Box(np.array(pos_choice), size, 0, np.random.randint(1, 10))
            # Check that the candidate's bottom face fits within the current available top tiles.
            if candidate_box.bottom_face() <= self.zone0_top:
                self.boxes.append(candidate_box)
                self.add_box_to_zone(candidate_box)

        # if tries == max_tries:
        #     print(f"Could not generate a valid initial state with {n_boxes} boxes")
        #     print(f"Generated {len(self.boxes)} boxes instead")

        self.t = 0
        self.valid_actions = []
        
        # Return observation
        state_3d = self.state_3d()  # returns [zone0_dense, zone1_dense]
        obs = {
            "state_zone0": np.expand_dims(state_3d[0].astype(np.float32), axis=0),
            "state_zone1": np.expand_dims(state_3d[1].astype(np.float32), axis=0)
        }
        return obs

    # Helper: return the index of the box with the given position and zone.
    def get_box_index(self, pos, zone):
        for i, b in enumerate(self.boxes):
            if np.array_equal(b.pos, np.array(pos)) and b.zone == zone:
                return i
        return -1

    def set_boxes(self, boxes):
        """
        Given a list of Box objects, recalculates and returns the available top positions
        for zone 0 and zone 1.
        
        For each zone, it starts with every tile (with z=0) as a candidate top position.
        Then, for each box in that zone, it adds the points of its top face and removes 
        the points of its bottom face.
        
        Args:
            boxes (list): List of Box objects.
            
        Returns:
            None
        """
        # Initialize candidate top positions for each zone.
        self.zone0_top = {(x, y, 0) for x in range(ZONE0[0]) for y in range(ZONE0[1])}
        self.zone1_top = {(x, y, 0) for x in range(ZONE1[0]) for y in range(ZONE1[1])}
        
        for box in boxes:
            self.add_box_to_zone(box)
        
        self.boxes = boxes

    # ==========================================================================
    #                               MDP Functions
    # ==========================================================================
    def actions(self):
        if self.valid_actions:
            return self.valid_actions
        actions = []
        # Consider only boxes in zone 0 that are not blocked.
        for box in self.boxes:
            if not (box.zone == 0 and box.top_face() <= self.zone0_top):
                continue
            p = box.pos
            s = box.size
            # Try moving the box to every available top position in zone 1.
            for possible_end in self.zone1_top:
                new_box = Box(np.array(possible_end), s, 1, 0)
                # Check that the new box's bottom face fits in zone 1 and does not exceed the zone's height.
                if new_box.bottom_face() <= self.zone1_top and (np.array(possible_end) + s)[2] <= ZONE1[2]:
                    # BoxAction takes the source position and target position.
                    actions.append(BoxAction(tuple(p), possible_end))
        self.valid_actions = actions
        return actions

    def step(self, action):
        """
        Takes a discrete action and advances the environment.
        
        Args:
            action (int): The discrete action index.
        
        Returns:
            obs (dict): The next state observation as a dict with keys "state_zone0" and "state_zone1".
            reward (float): The reward after taking the action.
            done (bool): True if the episode has ended normally.
            truncated (bool): True if the episode was truncated (for invalid actions or timeout).
            info (dict): Additional information.
        """
        if action not in self._action_map.keys():
            raise ValueError("Action index out of range.")
        chosen_action = self._action_map[action]

        valid = any(chosen_action == va for va in self.actions())
        if not valid:
            penalty = -1
            state_3d = self.state_3d()
            obs = {
                "state_zone0": np.expand_dims(state_3d[0].astype(np.float32), axis=0),
                "state_zone1": np.expand_dims(state_3d[1].astype(np.float32), axis=0)
            }
            # Invalid action - truncated but not terminated
            return obs, penalty, False, True, {"invalid_action": True}

        reward = self.apply_action(chosen_action)
        state_3d = self.state_3d()
        obs = {
            "state_zone0": np.expand_dims(state_3d[0].astype(np.float32), axis=0),
            "state_zone1": np.expand_dims(state_3d[1].astype(np.float32), axis=0)
        }
        
        # Check if episode should end
        terminated = (self.t == self.horizon or len(self.actions()) == 0)
        truncated = False
        info = {}
        
        return obs, reward, terminated, truncated, info

    def apply_action(self, action: BoxAction):
        """
        Applies a BoxAction to the environment.
        
        Args:
            action: The BoxAction to apply.
            
        Returns:
            float: The reward.
        """
        new_boxes = copy.deepcopy(self.boxes)
        idx = None
        # Find the box in zone 0 with the matching source position.
        for i, b in enumerate(new_boxes):
            if np.array_equal(b.pos, np.array(action.pos_from)) and b.zone == 0:
                idx = i
                break
        if idx is None:
            raise ValueError("No box found at the source position in zone 0.")
        box = new_boxes[idx]
        # Update the zone occupancy: remove the box from its current position.
        self.remove_box_from_zone(box)
        # Move the box to the new position in zone 1.
        box.pos = np.array(action.pos_to)
        box.zone = 1
        self.add_box_to_zone(box)
        self.boxes = new_boxes
        self.valid_actions = []  # Invalidate the current actions.
        self.t += 1
        return self.reward()

    def reward(self):
        # Use dense reward (proportional occupancy) when terminal or no actions available.
        if self.t == self.horizon or len(self.actions()) == 0:
            z0, z1 = self.state_3d()
            return np.sum(z1)
        else:
            return 0  # or use sparse rewards

    # ==========================================================================
    #                           RL Interface
    # ==========================================================================
    def state_1d(self):
        # Fill the state to a fixed length (one box per tile in zone 0).
        max_boxes = np.prod(ZONE0)
        boxes_copy = self.boxes.copy()
        while len(boxes_copy) < max_boxes:
            # Create a dummy box (using zeros and a zone of -1) as placeholder.
            boxes_copy.append(Box(np.zeros(3, dtype=int), np.zeros(3, dtype=int), -1, 0))
        state = np.zeros(max_boxes * BOX_DIM)
        for i, box in enumerate(boxes_copy):
            state[i * BOX_DIM : (i + 1) * BOX_DIM] = box.array_rep()
        return boxes_copy

    def state_3d(self):
        zone0_dense = zone0_dense_cpy()
        zone1_dense = zone1_dense_cpy()
        for i, box in enumerate(self.boxes):
            p = box.pos
            s = tuple(box.size)
            for offset in np.ndindex(s):
                coord = tuple(np.add(p, offset))
                if box.zone == 0:
                    zone0_dense[coord] = box.val_density()
                else:
                    zone1_dense[coord] = box.val_density()
        return [zone0_dense, zone1_dense]

    def action_1d(self, action):
        key = (*action.pos_from, *action.pos_to)
        if key not in self._action_1d_map:
            idx = 0
            for p_from in np.ndindex(ZONE0):
                for p_to in np.ndindex(ZONE1):
                    self._action_1d_map[(*p_from, *p_to)] = idx
                    idx += 1
        return self._action_1d_map[key]


    def action_3d(self, action):
        """
        Returns the 3D action representation as a pair of dense arrays for zone0 and zone1.
        If a box is not found at the source position in zone 0, a default (canonical) representation
        is generated assuming a box of size (1, 1, 1) with fixed density values.
        """
        p_from = np.array(action.pos_from)
        p_to = np.array(action.pos_to)
        idx = self.get_box_index(p_from, 0)
        if idx == -1:
            # Use a default canonical box size and densities.
            s = (1, 1, 1)
            density_zone0 = -1.0
            density_zone1 = 1.0
        else:
            box = self.boxes[idx]
            s = tuple(box.size)
            density_zone0 = -box.val_density()
            density_zone1 = box.val_density()
        zone0_dense = zone0_dense_cpy()
        zone1_dense = zone1_dense_cpy()
        from src.Constants import ZONE0, ZONE1
        for offset in np.ndindex(s):
            idx0 = tuple(np.add(p_from, offset))
            idx1 = tuple(np.add(p_to, offset))
            if all(0 <= idx0[i] < ZONE0[i] for i in range(3)):
                zone0_dense[idx0] = density_zone0
            if all(0 <= idx1[i] < ZONE1[i] for i in range(3)):
                zone1_dense[idx1] = density_zone1
        return [zone0_dense, zone1_dense]

    # ==========================================================================
    #                           Utility Functions
    # ==========================================================================
    def occupancy(self):
        total = sum(np.prod(box.size) for box in self.boxes if box.zone == 1)
        return total / np.prod(ZONE1)

    def add_box_to_zone(self, box):
        if box.zone == 0:
            self.zone0_top = self.zone0_top.union(box.top_face())
            self.zone0_top -= box.bottom_face()
        else:
            self.zone1_top = self.zone1_top.union(box.top_face())
            self.zone1_top -= box.bottom_face()

    def remove_box_from_zone(self, box):
        if box.zone == 0:
            self.zone0_top = self.zone0_top.union(box.bottom_face())
            self.zone0_top -= box.top_face()
        else:
            self.zone1_top = self.zone1_top.union(box.bottom_face())
            self.zone1_top -= box.top_face()

    def render(self, mode="human"):
        if mode == "human":
            self.visualize_scene()
        else:
            raise NotImplementedError(f"Render mode {mode} is not supported.")

    def close(self):
        plt.close("all")

    def visualize_scene(self):
        """
        Displays 3D views of both zones with:
        - The zone's bounding box.
        - Each box drawn as a filled cuboid with face colors corresponding to value density.
        - (Optionally) the wireframe around each box.
        - A color bar showing the mapping of value densities.
        """
        fig = plt.figure(figsize=(16, 8))

        # Create subplots for zone 0 and zone 1.
        ax0 = fig.add_subplot(121, projection="3d")
        ax0.set_title("Zone 0")
        ax0.set_xlim(0, ZONE0[0])
        ax0.set_ylim(0, ZONE0[1])
        ax0.set_zlim(0, ZONE0[2])

        ax1 = fig.add_subplot(122, projection="3d")
        ax1.set_title("Zone 1")
        ax1.set_xlim(0, ZONE1[0])
        ax1.set_ylim(0, ZONE1[1])
        ax1.set_zlim(0, ZONE1[2])

        def draw_zone_bounding_box(ax, zone_dims, color="black"):
            dx, dy, dz = zone_dims
            vertices = np.array([
                [0, 0, 0],
                [dx, 0, 0],
                [dx, dy, 0],
                [0, dy, 0],
                [0, 0, dz],
                [dx, 0, dz],
                [dx, dy, dz],
                [0, dy, dz]
            ])
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
                (4, 5), (5, 6), (6, 7), (7, 4),  # top
                (0, 4), (1, 5), (2, 6), (3, 7)   # vertical
            ]
            for i, j in edges:
                ax.plot([vertices[i][0], vertices[j][0]],
                        [vertices[i][1], vertices[j][1]],
                        [vertices[i][2], vertices[j][2]],
                        color=color, linewidth=2)

        # Draw zone boundaries.
        draw_zone_bounding_box(ax0, ZONE0, color="black")
        draw_zone_bounding_box(ax1, ZONE1, color="black")

        # Compute maximum value density for color mapping.
        max_density = max(box.val_density() for box in self.boxes) if self.boxes else 1
        norm = mcolors.Normalize(vmin=0, vmax=max_density)
        sm = cm.ScalarMappable(norm=norm, cmap="coolwarm")

        # Draw each box with a face color determined by its value density.
        for box in self.boxes:
            pos = box.pos
            size = box.size
            x, y, z = pos
            dx, dy, dz = size

            # Compute the 8 vertices for the box.
            vertices = np.array([
                [x, y, z],
                [x + dx, y, z],
                [x + dx, y + dy, z],
                [x, y + dy, z],
                [x, y, z + dz],
                [x + dx, y, z + dz],
                [x + dx, y + dy, z + dz],
                [x, y + dy, z + dz]
            ])

            # Define the 6 faces.
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
                [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
                [vertices[3], vertices[0], vertices[4], vertices[7]],  # left
            ]

            # Use the colormap to map the box's value density to an RGBA color.
            face_color = sm.to_rgba(box.val_density())

            # Choose the subplot based on the box's zone.
            ax = ax0 if box.zone == 0 else ax1

            poly3d = Poly3DCollection(faces, facecolors=face_color, edgecolors="none", alpha=0.7)
            ax.add_collection3d(poly3d)

        # Create a color bar to show the mapping of value densities.
        cbar = fig.colorbar(sm, ax=[ax0, ax1], fraction=0.02, pad=0.04)
        cbar.set_label("Value Density")

        # Set labels and view angle.
        for ax in [ax0, ax1]:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.view_init(elev=20, azim=30)

        plt.show()