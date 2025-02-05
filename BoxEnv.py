import numpy as np
import copy

DIM = 3
STATE_DIM = 2*DIM + 1
REMOVE_DIR = 2

def setup(remove_ax):
    global REMOVE_DIR
    REMOVE_DIR = ord(remove_ax) - ord('x')

class Box:
    """ Utility class."""

    @staticmethod
    def pos(box):
        return box[:DIM]
    
    @staticmethod
    def set_pos(box, pos):
        box[:DIM] = pos
    
    @staticmethod
    def size(box):
        return box[DIM:2*DIM]
    
    @staticmethod
    def set_size(box, size):
        box[DIM:2*DIM] = size
    
    @staticmethod
    def zone(box):
        return box[2*DIM]
    
    @staticmethod
    def set_zone(box, zone):
        box[2*DIM] = zone
    
    @staticmethod
    def make(pos, size, zone):
        return np.concatenate((pos, size, [zone]))
    
    @staticmethod
    def get_box(state, idx):
        """ Returns the box at the given index in the state vector. """
        idx *= STATE_DIM
        return state[idx:idx+STATE_DIM]
    
    @staticmethod
    def set_box(state, idx, box):
        """ Sets the box at the given index in the state vector. """
        idx *= STATE_DIM
        state[idx:idx+STATE_DIM] = box

    @staticmethod
    def boxes_from_state(state):
        """ Extracts a list of boxes from a state vector. """
        boxes = []
        for i in range(0, len(state)-1, STATE_DIM):
            boxes.append(state[i:i + STATE_DIM])
        return boxes

    @staticmethod
    def state_from_boxes(boxes, t=0):
        """ Creates a state vector from a list of boxes and a time step. """
        state = np.zeros(len(boxes) * (STATE_DIM) + 1, dtype=int)
        for i, box in enumerate(boxes):
            Box.set_box(state, i, box)
        state[-1] = t
        return state
    
    @staticmethod
    def box_idx(state, pos, zone):
        """ Returns the index of the box with the given position and zone. """
        boxes = Box.boxes_from_state(state)
        for i, box in enumerate(boxes):
            if (Box.pos(box) == pos).all() and Box.zone(box) == zone:
                return i
        return -1
    
    @staticmethod
    def to_str(state):
        boxes = Box.boxes_from_state(state)
        return str([f"p={Box.pos(box)}, s={Box.size(box)}, z={Box.zone(box)}" for box in boxes]) + f", t={state[-1]}"
    
    @staticmethod
    def key_from_pair(box1, box2):
        """ Returns a hashable key for a pair of boxes."""
        box1_first = False
        for i in range(STATE_DIM):
            box1_first = box1[i] < box2[i]
            if box1[i] != box2[i]:
                break
        it = iter([box1, box2]) if box1_first else iter([box2, box1])
        key = []
        for box in it:
            key.extend(list(box))
        return tuple(key)

    @staticmethod
    def in_front(box1, box2):
        """ Returns True if box1 is in front of box2, False otherwise. """
        return Box.pos(box1)[REMOVE_DIR] > Box.pos(box2)[REMOVE_DIR]

    @staticmethod
    def null_box():
        """ Creates a box that is not valid for placeholding."""
        return Box.make(np.zeros(DIM), np.zeros(DIM), -1)

    @staticmethod
    def top_and_bottom(box):
        """ Computes the top and bottom points of the box. """
        bottom_points = set()
        top_points = set()
        pos = Box.pos(box)
        size = Box.size(box)
        for i in range(size[0]+1):
            for j in range(size[1]+1):
                bottom_points.add(tuple(pos + np.array([i, j, 0])))
                top_points.add(tuple(pos + np.array([i, j, size[2]])))
        return top_points, bottom_points


    
class MoveBox:
    """ Encodes a box movement action."""
    def __init__(self, pos_from, pos_to, zone_from, zone_to):
        self.pos_from = pos_from
        self.pos_to = pos_to
        self.zone_from = zone_from
        self.zone_to = zone_to

    def is_null(self):
        return np.array_equal(self.pos_from, self.pos_to) and self.zone_from == self.zone_to

    def __str__(self):
        return f"({self.pos_from} -> {self.pos_to}, {self.zone_from} -> {self.zone_to})"

    def __eq__(self, other):
        return np.array_equal(self.pos_from, other.pos_from) \
                and np.array_equal(self.pos_to, other.pos_to) \
                and self.zone_from == other.zone_from \
                and self.zone_to == other.zone_to

class BoxMoveEnvironment:
    """
    Gravity is in the negative z direction.
    """

    def __init__(self, zone_sizes, horizon=100, gamma=1):
        self.zone_sizes = zone_sizes
        self.n_zones = len(zone_sizes)
        self.horizon = horizon
        self.gamma = gamma

    def actions(self, state: np.array):
        """
        Computes a list of possible actions from the current state. Please see
        the MoveBox class for more information on the action representation.

        Args:
            state: the current state of the environment.

        Returns:
            list: a list of possible actions from the current state.
        """
        actions = []
        collisions = set()
        all_boxes = Box.boxes_from_state(state)
        for i in range(self.n_zones):
            zone_boxes = [box for box in all_boxes if Box.zone(box) == i]
            collisions |= self._collision_pairs_2d(zone_boxes, zone_boxes, proj_dim=REMOVE_DIR)
        unmovable_boxes = set()
        for pair in collisions:
            box1 = pair[:STATE_DIM]
            box2 = pair[STATE_DIM:]
            unmovable = box2 if Box.in_front(box1, box2) else box1
            unmovable_boxes.add(unmovable)
        moveable_boxes = set([tuple(b) for b in all_boxes]) - unmovable_boxes
        moveable_boxes = [b for b in moveable_boxes if Box.zone(b) == 0]
        
        for box in moveable_boxes:
            # consider the action of moving a box to all possible positions
            start_pos = Box.pos(box)
            start_zone = Box.zone(box)
            # for zone in range(self.n_zones): # OR we can just consider the target zone
            for zone in [1]:
                box_removed_state = copy.deepcopy(state)
                Box.set_box(box_removed_state, Box.box_idx(state, start_pos, start_zone), Box.null_box())
                for possible_end in self._zone_top_bottom(box_removed_state, zone)[0]:
                    # take the action, and if valid, add it to the list of actions
                    action = MoveBox(start_pos, possible_end, start_zone, zone)
                    if self._is_valid_action(state, action):
                        actions.append(copy.deepcopy(action))
        return actions

    def step(self, state: np.array, action: MoveBox):
        """
        Takes a step in the simulator.

        Args:
            state (np.array): the current state of the environment.
            action (MoveBox): the action to take.
        
        Returns:
            tuple: a tuple containing the next state of the environment, the
            reward received from taking the action, and a boolean indicating
            whether the episode has ended.
        """
        state = self.transition(state, action)
        state[-1] += 1
        if state[-1] == self.horizon:
            return state, self.reward(), True
        else:
            return state, 0, False

    def transition(self, curr_state, action):
        """Computes the next state of the environment given an action.

        Args:
            action (MoveBox): the action to take.

        Returns:
            np.array: the next state of the environment.
        """
        state = copy.deepcopy(curr_state)
        idx = Box.box_idx(state, action.pos_from, action.zone_from)
        box_state = Box.get_box(state, idx)
        Box.set_pos(box_state, action.pos_to)
        Box.set_zone(box_state, action.zone_to)
        Box.set_box(state, idx, box_state)
        return state


    def reward(self):
        return self.occupancy()

    
    def common_state_vec(self, state):
        boxes = Box.boxes_from_state(state)
        t = state[-1]
        max_boxes = self.zone_sizes[0][0] * self.zone_sizes[0][1] * self.zone_sizes[0][2]
        null_box = Box.null_box()
        common_state = np.array([null_box for _ in range(max_boxes)], dtype=int)
        for i, box in enumerate(boxes):
            Box.set_box(common_state, i, box)
        common_state[-1] = t
        return common_state


    def _is_valid_action(self, state: np.array, action):
        """
        Tests if an action is valid.

        Args:
            state (np.array): the current state of the environment.
            action (MoveBox): the action to test.

        Returns:
            bool: True if the action is valid, False otherwise.
        """
        new_state = self.transition(state, action)
        orig_box = Box.get_box(state, Box.box_idx(state, action.pos_from, action.zone_from))
        new_box = Box.make(action.pos_to, Box.size(orig_box), action.zone_to)
        new_boxes = Box.boxes_from_state(new_state)
        # check if the action is null
        if action.is_null():
            return False
        # check if the new state is valid
        if not self._is_valid_state(new_state):
            return False
        # check if the target position can be reached
        col_pairs_2d = self._collision_pairs_2d([new_box], new_boxes, proj_dim=REMOVE_DIR)
        for pair in col_pairs_2d:
            box1 = pair[:STATE_DIM]
            box2 = pair[STATE_DIM:]
            if (box1 == new_box).all() and Box.in_front(box2, box1):
                return False
            elif (box2 == new_box).all() and Box.in_front(box1, box2):
                return False
        return True


    def _collision_pairs_1d(self, group1, group2, dim):
        """ Computes the pairs of boxes that collide on a given axis. """
        groups1d = [[], []]
        i = 0
        for g in [group1, group2]:
            for box in g:
                pos = Box.pos(box)
                size = Box.size(box)
                groups1d[i].append((pos[dim], (pos + size)[dim], box))
            groups1d[i].sort(key=lambda x: x[0])
            i += 1
        pairs = set()
        for i, box1 in enumerate(groups1d[0]):
            for j, box2 in enumerate(groups1d[1]):
                if (box1[2] == box2[2]).all() and (i==j):   # ensure they're not the same box
                    continue
                a, b, c, d = box1[0], box1[1], box2[0], box2[1]
                if a < c < b or a < d < b or a == c or b == d:
                    pairs.add(Box.key_from_pair(box1[2], box2[2]))
        return pairs

    def _collision_pairs_2d(self, group1, group2, proj_dim):
        """ Computes the pairs of boxes that collide on a given plane defined by the normal direction in proj_dim. """
        pairs = set()
        added = False
        for i in range(DIM):
            if i == proj_dim:
                continue
            if not added:
                pairs = self._collision_pairs_1d(group1, group2, i)
                added = True
            else:
                pairs &= self._collision_pairs_1d(group1, group2, i)
        return pairs
    

    def _collision_pairs_3d(self, group1, group2):
        return self._collision_pairs_2d(group1, group2, proj_dim=-1)
    

    def _is_valid_state(self, state):
        """Checks if the current state is valid."""

        boxes = Box.boxes_from_state(state)

        for zone in range(self.n_zones):
            zone_size = self.zone_sizes[zone]
            boxes_zone = [box for box in boxes if Box.zone(box) == zone]

            # check out-of-bound
            for box in boxes_zone:
                pos = Box.pos(box)
                size = Box.size(box)
                if (pos < 0).any() or (pos + size > zone_size).any():
                    return False

            # check collisions
            if len(self._collision_pairs_3d(boxes_zone, boxes_zone)) > 0:
                return False

            # check support
            top, _ = self._zone_top_bottom(state, zone)
            for box in boxes_zone:
                _, box_bottom = Box.top_and_bottom(box)
                if not box_bottom.issubset(top):
                    return False
        return True

    def _zone_top_bottom(self, state: np.array, zone):
        """
        Computes the top and bottom points of the zone.
        
        Args:
            state (np.array): the current state of the environment.
            zone (int): the zone to consider.
        
        Returns:
            tuple: a tuple of two sets containing the top (floor and box tops) and bottom 
            (ceiling and box bottoms) points of the zone.
        """
        boxes = [b for b in Box.boxes_from_state(state) if Box.zone(b) == zone]
        top = set()
        bottom = set()
        # first add the floor and ceiling
        for i in range(self.zone_sizes[zone][0]+1):
            for j in range(self.zone_sizes[zone][1]+1):
                top.add((i, j, 0))
                bottom.add((i, j, self.zone_sizes[zone][2]))
        # then add the boxes
        for box in boxes:
            top_points, bottom_points = Box.top_and_bottom(box)
            top |= top_points
            bottom |= bottom_points
        return top, bottom


    def occupancy(self, state: np.array, zone=1):
        """Computes the spacial occupancy of the target zone.

        Args:
            state (np.array): the current state of the environment.

        Returns:
            float: the fraction of the target zone that is occupied by boxes.
        """
        occupancy = 0
        for box in Box.boxes_from_state(state):
            if Box.zone(box) == 1:
                occupancy += np.prod(box.size)
        occupancy /= np.prod(self.zone_sizes[zone])
        return occupancy
    

    def random_initial_state(env, n_boxes, zone=0):
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