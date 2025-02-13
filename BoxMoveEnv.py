import numpy as np
import random

from BoxAction import BoxAction
from Constants import ZONE0, ZONE1
import Box
import copy

class BoxMoveEnv:
    def __init__(self, horizon=100, gamma=1, n_boxes=5):
        self.horizon = horizon
        self.gamma = gamma
        self.state = None
        self.zone0_top = set()
        self.zone1_top = set()
        self.valid_actions = []
        self.n_boxes = n_boxes
        self.reset(n_boxes=n_boxes)

    def reset(self, n_boxes=0):
        if n_boxes == 0:
            n_boxes = self.n_boxes
        boxes = []
        tries = 0
        max_tries = 1000
        self.zone0_top = set()
        self.zone1_top = set()
        for x in range(ZONE0[0]):
            for y in range(ZONE0[1]):
                    self.zone0_top.add((x, y, 0))
        for x in range(ZONE1[0]):
            for y in range(ZONE1[1]):
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
            raise ValueError(f"Could not generate a valid initial state with {n_boxes} boxes")
    
    def actions(self):
        # First compute all the movable boxes (i.e., those not blocked by other boxes)
        free_boxes = []
        for box in Box.boxes_from_state(self.state):
            if Box.zone(box) == 0 and Box.top_face(box) <= self.zone0_top:
                free_boxes.append(box)

        # Then consider the action of moving a box to "top" points in zone 1
        actions = []
        for box in free_boxes:
            p = Box.pos(box)
            s = Box.size(box)
            for possible_end in self.zone1_top:
                box_new = Box.make(possible_end, s, 1)
                # the box is settled, and does not exceed the zone1 boundary
                if (Box.bottom_face(box_new) <= self.zone1_top) \
                    and (p+s)[2] < ZONE1[2]:
                    actions.append(BoxAction(p, possible_end, 0, 1))
        self.valid_actions = actions
        return self.valid_actions
    

    def step(self, action: BoxAction):
        state = copy.deepcopy(self.state)
        idx = Box.box_idx(state, action.pos_from, action.zone_from)
        box_state = Box.access_from_state(state, idx)
        self.remove_box_from_zone(box_state)

        Box.pos(box_state, action.pos_to)
        Box.zone(box_state, action.zone_to)
        Box.access_from_state(state, idx, box_state)
        self.add_box_to_zone(box_state)

        self.state = state
        self.valid_actions = self.actions()
        return self.reward()


    def reward(self):
        return self.occupancy() if self.state[-1] == self.horizon or len(self.actions()) ==0 \
                                else 0
    

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