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
            print(f"Generated {len(boxes)} boxes instead")
        

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
                if (Box.bottom_face(box_new) <= self.zone1_top) and np.add(possible_end, s)[2] <= ZONE1[2]:
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
                if zone == 0:
                    zone0_dense[coord] = box_idx + 1
                else:
                    zone1_dense[coord] = box_idx + 1
        
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
        # Use zone 0 for the source box (since boxes are moved from zone 0 to zone 1)
        box_idx = Box.box_idx(self.state, p_from, 0)
        box = Box.access_from_state(self.state, box_idx)
        s = tuple(Box.size(box))  # Ensure s is a tuple of ints
        zone0_dense = zone0_dense_cpy()
        zone1_dense = zone1_dense_cpy()
        for offset in np.ndindex(s):
            # Compute new coordinates by elementwise addition and convert to tuple for indexing.
            zone0_dense[tuple(np.add(p_from, offset))] = box_idx + 1
            zone1_dense[tuple(np.add(p_to, offset))] = box_idx + 1
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
            self.zone1_top = self.zone1_top.union(Box.top_face(box))
            self.zone1_top -= Box.bottom_face(box)

    def remove_box_from_zone(self, box):
        if Box.zone(box) == 0:
            self.zone0_top = self.zone0_top.union(Box.bottom_face(box))
            self.zone0_top -= Box.top_face(box)
        else:
            self.zone1_top = self.zone1_top.union(Box.bottom_face(box))
            self.zone1_top -= Box.top_face(box)

    def visualize_scene(self):
        """
        Creates two 3D graphs (one for each zone) displaying all boxes in the environment.
        In each subplot, it draws:
        - The bounding box of the zone.
        - Each box as a filled cuboid.
        - The bounding box (wireframe) of each box in a color specific to the zone.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import numpy as np
        from Constants import ZONE0, ZONE1

        fig = plt.figure(figsize=(16, 8))
        
        # Create subplots for zone 0 and zone 1.
        ax0 = fig.add_subplot(121, projection='3d')
        ax0.set_title("Zone 0")
        ax0.set_xlim(0, ZONE0[0])
        ax0.set_ylim(0, ZONE0[1])
        ax0.set_zlim(0, ZONE0[2])
        
        ax1 = fig.add_subplot(122, projection='3d')
        ax1.set_title("Zone 1")
        ax1.set_xlim(0, ZONE1[0])
        ax1.set_ylim(0, ZONE1[1])
        ax1.set_zlim(0, ZONE1[2])
        
        # Helper function: draw the bounding box (wireframe) for a zone.
        def draw_zone_bounding_box(ax, zone_dims, color='black'):
            dx, dy, dz = zone_dims
            # Define the 8 vertices for the zone cuboid.
            vertices = np.array([
                [0, 0, 0],
                [dx, 0, 0],
                [dx, dy, 0],
                [0, dy, 0],
                [0, 0, dz],
                [dx, 0, dz],
                [dx, dy, dz],
                [0, dy, dz],
            ])
            # Define all 12 edges by index pairs.
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

        # Draw the zone boundaries.
        draw_zone_bounding_box(ax0, ZONE0, color='black')
        draw_zone_bounding_box(ax1, ZONE1, color='black')
        
        # Get all boxes from the current state.
        boxes = Box.boxes_from_state(self.state)
        
        for box in boxes:
            pos = Box.pos(box)      # (x, y, z)
            size = Box.size(box)    # (dx, dy, dz)
            zone = Box.zone(box)    # 0 or 1
            # You can use the box index for a bit of color variation.
            box_idx = Box.box_idx(self.state, pos, zone)
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
            
            # Define the 6 faces for the box.
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
                [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
                [vertices[3], vertices[0], vertices[4], vertices[7]]   # left
            ]
            
            # Choose subplot and wireframe color based on zone.
            if zone == 0:
                # For zone 0, use a blue-based color.
                face_color = np.array([0, 0, 1 - box_idx / (len(boxes) + 1) * 0.5])
                wire_color = 'magenta'
                ax = ax0
            else:
                # For zone 1, use a red-based color.
                face_color = np.array([1 - box_idx / (len(boxes) + 1) * 0.5, 0, 0])
                wire_color = 'red'
                ax = ax1
            
            # Draw filled faces for the box (semi-transparent).
            poly3d = Poly3DCollection(faces, facecolors=face_color, edgecolors='none', alpha=0.7)
            ax.add_collection3d(poly3d)
            
            # Now draw the bounding box (wireframe) for the box.
            # edges = [
            #     (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            #     (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            #     (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
            # ]
            # for i, j in edges:
            #     ax.plot([vertices[i][0], vertices[j][0]],
            #             [vertices[i][1], vertices[j][1]],
            #             [vertices[i][2], vertices[j][2]],
            #             color=wire_color, linewidth=2)
        
        # Set axis labels and a consistent view angle.
        for ax in [ax0, ax1]:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=20, azim=30)
        
        plt.tight_layout()
        plt.show()
