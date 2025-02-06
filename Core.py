import numpy as np
import copy
import Box
from BoxAction import BoxAction
from Constants import DIM, STATE_DIM, REMOVE_DIR, ZONE_SIZES, ZONE0, ZONE1


def actions(state: np.ndarray):
    """
    Computes a list of possible actions from the current state. Please see
    the BoxAction class for more information on the action representation.

    Args:
        state: the current state of the environment.

    Returns:
        list: a list of possible actions from the current state.
    """
    # First compute all the movable boxes (i.e., those not blocked by other boxes)
    collisions = set()
    all_boxes = Box.boxes_from_state(state)
    zone_boxes = [box for box in all_boxes if Box.zone(box) == 0]
    collisions |= _collision_pairs_2d(zone_boxes, zone_boxes, proj_dim=REMOVE_DIR)
    unmovable_boxes = set()
    for pair in collisions:
        box1 = pair[:STATE_DIM]
        box2 = pair[STATE_DIM:]
        unmovable = box2 if Box.in_front(box1, box2) else box1
        unmovable_boxes.add(unmovable)
    moveable_boxes = set([tuple(b) for b in all_boxes]) - unmovable_boxes
    moveable_boxes = [b for b in moveable_boxes if Box.zone(b) == 0]
    
    # Then consider the action of moving a box to "top" points in zone 1
    actions = []
    for box in moveable_boxes:
        start_pos = Box.pos(box)
        start_zone = Box.zone(box)
        zone = 1
        box_removed_state = copy.deepcopy(state)
        Box.access_from_state(box_removed_state, Box.box_idx(state, start_pos, start_zone), Box.null_box())
        for possible_end in _zone_top_bottom(box_removed_state, zone)[0]:
            # take the action, and if valid, add it to the list of actions
            action = BoxAction(start_pos, possible_end, start_zone, zone)
            if _is_valid_action(state, action):
                actions.append(copy.deepcopy(action))
    return actions


def transition(curr_state, action, step_time=False):
    """Computes the next state of the environment given an action.

    Args:
        action (BoxAction): the action to take.

    Returns:
        np.ndarray: the next state of the environment.
    """
    state = copy.deepcopy(curr_state)
    idx = Box.box_idx(state, action.pos_from, action.zone_from)
    box_state = Box.access_from_state(state, idx)
    Box.pos(box_state, action.pos_to)
    Box.zone(box_state, action.zone_to)
    Box.access_from_state(state, idx, box_state)
    if step_time:
        state[-1] += 1
    return state

def _is_valid_action(state: np.ndarray, action):
    """
    Tests if an action is valid.

    Args:
        state (np.ndarray): the current state of the environment.
        action (BoxAction): the action to test.

    Returns:
        bool: True if the action is valid, False otherwise.
    """
    new_state = transition(state, action)
    orig_box = Box.access_from_state(state, Box.box_idx(state, action.pos_from, action.zone_from))
    new_box = Box.make(action.pos_to, Box.size(orig_box), action.zone_to)
    new_boxes = Box.boxes_from_state(new_state)
    # check if the action is null
    if action.is_null():
        return False
    # check if the new state is valid
    if not _is_valid_state(new_state):
        return False
    # check if the target position can be reached
    col_pairs_2d = _collision_pairs_2d([new_box], new_boxes, proj_dim=REMOVE_DIR)
    for pair in col_pairs_2d:
        box1 = pair[:STATE_DIM]
        box2 = pair[STATE_DIM:]
        if (box1 == new_box).all() and Box.in_front(box2, box1):
            return False
        elif (box2 == new_box).all() and Box.in_front(box1, box2):
            return False
    return True


def _collision_pairs_1d(group1, group2, dim):
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

def _collision_pairs_2d(group1, group2, proj_dim):
    """ Computes the pairs of boxes that collide on a given plane defined by the normal direction in proj_dim. """
    pairs = set()
    added = False
    for i in range(DIM):
        if i == proj_dim:
            continue
        if not added:
            pairs = _collision_pairs_1d(group1, group2, i)
            added = True
        else:
            pairs &= _collision_pairs_1d(group1, group2, i)
    return pairs


def _collision_pairs_3d(group1, group2):
    return _collision_pairs_2d(group1, group2, proj_dim=-1)


def _is_valid_state(state):
    """Checks if the current state is valid."""

    boxes = Box.boxes_from_state(state)

    for zone in [0, 1]:
        zone_size = ZONE_SIZES[zone]
        boxes_zone = [box for box in boxes if Box.zone(box) == zone]

        # check out-of-bound
        for box in boxes_zone:
            pos = Box.pos(box)
            size = Box.size(box)
            if (pos < 0).any() or (pos + size > zone_size).any():
                return False

        # check collisions
        if len(_collision_pairs_3d(boxes_zone, boxes_zone)) > 0:
            return False

        # check support
        top, _ = _zone_top_bottom(state, zone)
        for box in boxes_zone:
            _, box_bottom = Box.top_and_bottom(box)
            if not box_bottom.issubset(top):
                return False
    return True

def _zone_top_bottom(state: np.ndarray, zone):
    """
    Computes the top and bottom points of the zone.
    
    Args:
        state (np.ndarray): the current state of the environment.
        zone (int): the zone to consider.
    
    Returns:
        tuple: a tuple of two sets containing the top (floor and box tops) and bottom 
        (ceiling and box bottoms) points of the zone.
    """
    boxes = [b for b in Box.boxes_from_state(state) if Box.zone(b) == zone]
    top = set()
    bottom = set()
    # first add the floor and ceiling
    for i in range(ZONE_SIZES[zone][0]+1):
        for j in range(ZONE_SIZES[zone][1]+1):
            top.add((i, j, 0))
            bottom.add((i, j, ZONE_SIZES[zone][2]))
    # then add the boxes
    for box in boxes:
        top_points, bottom_points = Box.top_and_bottom(box)
        top |= top_points
        bottom |= bottom_points
    return top, bottom


def occupancy(state: np.ndarray, zone=1):
    """Computes the spacial occupancy of the target zone.

    Args:
        state (np.ndarray): the current state of the environment.

    Returns:
        float: the fraction of the target zone that is occupied by boxes.
    """
    occupancy = 0
    for box in Box.boxes_from_state(state):
        if Box.zone(box) == 1:
            occupancy += np.prod(box.size)
    occupancy /= np.prod(ZONE_SIZES[zone])
    return occupancy


def random_initial_state(n_boxes, zone=0):
    """
    Randomly create a valid initial state (all boxes in zone 0).
    """
    boxes = []
    tries = 0
    max_tries = 500
    while len(boxes) < n_boxes and tries < max_tries:
        tries += 1
        x_size = np.random.randint(1, ZONE0[0])
        y_size = np.random.randint(1, ZONE0[1])
        z_size = np.random.randint(1, ZONE0[2])
        # Random position ensuring the box fits
        x_pos = np.random.randint(0, ZONE0[0] - x_size + 1)
        y_pos = np.random.randint(0, ZONE0[1] - y_size + 1)
        z_pos = np.random.randint(0, ZONE0[2] - z_size + 1)
        candidate_box = Box.make(
            p=[x_pos, y_pos, z_pos],
            s=[x_size, y_size, z_size],
            zone=zone
        )
        # Test if adding this new box is still a valid state
        test_boxes = boxes + [candidate_box]
        test_state = Box.state_from_boxes(test_boxes)
        if _is_valid_state(test_state):
            boxes.append(candidate_box)
    # Create the final state from whatever valid boxes we have
    return Box.state_from_boxes(boxes, t=0)


def all_actions():
    all_xns = []
    for from_pos in np.ndindex(ZONE0):
        for to_pos in np.ndindex(ZONE1):
            all_xns.append(BoxAction(from_pos, to_pos, 0, 1))
    return all_xns

def pad_boxes(state, n_boxes=None):
    """Pads the state with null boxes to reach n_boxes."""
    if n_boxes is None:
        n_boxes = np.prod(ZONE0)
    boxes = Box.boxes_from_state(state)
    while len(boxes) < n_boxes:
        boxes.append(Box.null_box())
    return Box.state_from_boxes(boxes, state[-1])