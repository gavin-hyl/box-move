import numpy as np
import random

from BoxAction import BoxAction
from Constants import ZONE0, ZONE1, zone0_dense_cpy, zone1_dense_cpy
import Box
import copy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


class BoxMoveEnv:
    # ==========================================================================
    #                               Initialization
    # ==========================================================================

    def __init__(self, horizon=100, gamma=1, n_boxes=5):
        self.horizon = horizon
        self.gamma = gamma
        self.state = None
        self.zone0_top = set()
        self.zone1_top = set()
        self.valid_actions = []
        self.n_boxes = n_boxes
        self.reset(n_boxes=n_boxes)
        self._action_1d_map = None

    def reset(self, n_boxes=0):
        if n_boxes == 0:
            n_boxes = self.n_boxes
        boxes = []
        tries = 0
        max_tries = 1000
        self.zone0_top = set()
        self.zone1_top = set()
        for i, zone_size in enumerate([ZONE0, ZONE1]):
            for x in range(zone_size[0]):
                for y in range(zone_size[1]):
                    if i == 0:
                        self.zone0_top.add((x, y, 0))
                    else:
                        self.zone1_top.add((x, y, 0))

        while len(boxes) < n_boxes and tries < max_tries:
            tries += 1
            pos = random.choice(list(self.zone0_top))
            if pos[2] >= ZONE0[2]:
                continue
            size = np.random.randint(1, np.array(ZONE0) - np.array(pos) + 1, size=3)
            candidate_box = Box.make(pos, size, 0)
            # Test if adding this new box is still a valid state
            if Box.bottom_face(candidate_box) <= self.zone0_top:
                boxes.append(candidate_box)
                self.state = Box.state_from_boxes(boxes, t=0)
                self.add_box_to_zone(candidate_box)

        if tries == max_tries:
            print(f"Could not generate a valid initial state with {n_boxes} boxes")
        

    # ==========================================================================
    #                               MDP Functions
    # ==========================================================================

    def actions(self):
        if self.valid_actions:
            return self.valid_actions
        # A transition is made, need to recompute the valid actions
        # First compute all the movable boxes (i.e., those not blocked by other boxes)
        actions = []
        for box in Box.boxes_from_state(self.state):
            if not (Box.zone(box) == 0 and Box.top_face(box) <= self.zone0_top):
                continue
            # consider the action of moving a box to "top" points in zone 1
            p = Box.pos(box)
            s = Box.size(box)
            for possible_end in self.zone1_top:
                box_new = Box.make(possible_end, s, 1)
                # the box is settled, and does not exceed the zone1 boundary
                if (Box.bottom_face(box_new) <= self.zone1_top) and (p + s)[2] <= ZONE1[
                    2
                ]:
                    actions.append(BoxAction(p, possible_end))
        self.valid_actions = actions
        return self.valid_actions

    def step(self, action: BoxAction):
        state = copy.deepcopy(self.state)
        idx = Box.box_idx(state, action.pos_from, 0)
        box_state = Box.access_from_state(state, idx)
        self.remove_box_from_zone(box_state)

        Box.pos(box_state, action.pos_to)
        Box.zone(box_state, 1)
        Box.access_from_state(state, idx, box_state)
        self.add_box_to_zone(box_state)

        self.state = state
        self.valid_actions = None
        return self.reward()

    def reward(self):
        return (
            self.occupancy()
            if self.state[-1] == self.horizon or len(self.actions()) == 0
            else 0
        )

    # ==========================================================================
    #                           RL Interface
    # ==========================================================================
    def state_1d(self):
        max_boxes = np.prod(ZONE0)
        boxes = Box.boxes_from_state(self.state)
        while len(boxes) < max_boxes:
            boxes.append(Box.null_box())
        return Box.state_from_boxes(boxes, t=self.state[-1])

    def state_3d(self):
        zone0_dense = zone0_dense_cpy()
        zone1_dense = zone1_dense_cpy()
        
        boxes = Box.boxes_from_state(self.state)
        
        for box in boxes:
            p = Box.pos(box)
            s = tuple(Box.size(box))
            zone = Box.zone(box)
            box_idx = Box.box_idx(self.state, p, zone)
            
            for offset in np.ndindex(s):
                coord = tuple(np.add(p, offset))
                zone0_dense[coord] = box_idx + 1
        
        return [zone0_dense, zone1_dense]

    def action_1d(self, action):
        if self._action_1d_map is not None:
            key = (*action.pos_from, *action.pos_to)
        idx = 0
        for p_from in np.ndindex(ZONE0):
            for p_to in np.ndindex(ZONE1):
                self._action_1d_map[(*p_from, *p_to)] = idx
                idx += 1
        return self._action_1d_map[key]

    def action_3d(self, action):
        p_from = action.pos_from
        p_to = action.pos_to
        box_idx = Box.box_idx(self.state, p_from, action.zone_from)
        box = Box.access_from_state(self.state, box_idx)
        s = Box.size(box)
        zone0_dense = zone0_dense_cpy()
        zone1_dense = zone1_dense_cpy()
        for point in np.ndindex(s):
            zone0_dense[p_from + point] = -1 * (box_idx + 1)
            zone1_dense[p_to + point] = box_idx + 1
        return [zone0_dense, zone1_dense]

    # ==========================================================================
    #                           Utility Functions
    # ==========================================================================

    def occupancy(self):
        return sum(
            [
                np.prod(Box.size(box))
                for box in Box.boxes_from_state(self.state)
                if Box.zone(box) == 1
            ]
        ) / np.prod(ZONE1)

    def add_box_to_zone(self, box):
        if Box.zone(box) == 0:
            self.zone0_top = self.zone0_top.union(Box.top_face(box))
            self.zone0_top -= Box.bottom_face(box)
        else:
            self.zone1_top += Box.top_face(box)
            self.zone1_top -= Box.bottom_face(box)

    def remove_box_from_zone(self, box):
        if Box.zone(box) == 0:
            self.zone0_top += Box.bottom_face(box)
            self.zone0_top -= Box.top_face(box)
        else:
            self.zone1_top += Box.bottom_face(box)
            self.zone1_top -= Box.top_face(box)

    def visualize_scene(self):
        """
        Creates two 3D graphs (one for each zone) displaying all boxes in the environment.
        """
        # Create a figure with two 3D subplots
        fig = plt.figure(figsize=(16, 8))
        
        # Zone 0 subplot
        ax0 = fig.add_subplot(121, projection='3d')
        ax0.set_title("Zone 0")
        # Set axis limits using ZONE0 dimensions.
        ax0.set_xlim(0, ZONE0[0])
        ax0.set_ylim(0, ZONE0[1])
        ax0.set_zlim(0, ZONE0[2])
        
        # Zone 1 subplot
        ax1 = fig.add_subplot(122, projection='3d')
        ax1.set_title("Zone 1")
        # Set axis limits using ZONE1 dimensions.
        ax1.set_xlim(0, ZONE1[0])
        ax1.set_ylim(0, ZONE1[1])
        ax1.set_zlim(0, ZONE1[2])
        
        # Get all boxes from the current state
        boxes = Box.boxes_from_state(self.state)
        
        # Iterate through each box and draw it in its corresponding zone subplot.
        for box in boxes:
            pos = Box.pos(box)      # (x, y, z) position of the box
            size = Box.size(box)    # (dx, dy, dz) dimensions of the box
            zone = Box.zone(box)    # Zone number: 0 or 1
            
            # For a unique identifier (optional) - may be used for coloring, etc.
            box_idx = Box.box_idx(self.state, pos, zone)
            
            # Unpack position and size for clarity.
            x, y, z = pos
            dx, dy, dz = size
            
            # Compute the eight vertices of the cuboid.
            vertices = np.array([
                [x,      y,      z],
                [x + dx, y,      z],
                [x + dx, y + dy, z],
                [x,      y + dy, z],
                [x,      y,      z + dz],
                [x + dx, y,      z + dz],
                [x + dx, y + dy, z + dz],
                [x,      y + dy, z + dz],
            ])
            
            # Define the six faces of the cuboid.
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
                [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
                [vertices[3], vertices[0], vertices[4], vertices[7]],  # left
            ]
            
            # Choose color: blue for zone 0, red for zone 1.
            color = 'blue' if zone == 0 else 'red'
            poly3d = Poly3DCollection(faces, facecolors=color, edgecolors='k', alpha=0.7)
            
            # Add the cuboid to the corresponding subplot.
            if zone == 0:
                ax0.add_collection3d(poly3d)
            else:
                ax1.add_collection3d(poly3d)
        
        # Set axis labels and a consistent view angle for both subplots.
        for ax in [ax0, ax1]:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=20, azim=30)
        
        plt.tight_layout()
        plt.show()