import numpy as np
from Constants import DIM, STATE_DIM, REMOVE_DIR

def pos(box, p=None):
    if p is not None:
        box[:DIM] = p
    return box[:DIM]

def size(box, size=None):
    if size is not None:
        box[DIM:2*DIM] = size
    return box[DIM:2*DIM]

def zone(box, zone=None):
    if zone is not None:
        box[2*DIM] = zone
    return box[2*DIM]

def make(p, s, zone):
    rep = np.zeros(STATE_DIM, dtype=int)
    for i, x in enumerate(p):
        rep[i] = x
    for i, x in enumerate(s):
        rep[i + DIM] = x
    rep[2*DIM] = zone
    return rep

def access_from_state(state, idx, new_box=None):
    """ Returns the box at the given index in the state vector. """
    idx *= STATE_DIM
    if new_box is not None:
        state[idx:idx + STATE_DIM] = new_box
    return state[idx:idx + STATE_DIM]


def boxes_from_state(state):
    """ Extracts a list of boxes from a state vector. """
    boxes = []
    for i in range(0, len(state)-1, STATE_DIM):
        boxes.append(state[i:i + STATE_DIM])
    return boxes


def state_from_boxes(boxes, t=0):
    """ Creates a state vector from a list of boxes and a time step. """
    state = np.zeros(len(boxes) * (STATE_DIM) + 1, dtype=int)
    for i, box in enumerate(boxes):
        access_from_state(state, i, box)
    state[-1] = t
    return state


def box_idx(state, p, z):
    """ Returns the index of the box with the given position and zone. """
    boxes = boxes_from_state(state)
    for i, box in enumerate(boxes):
        if (pos(box) == p).all() and zone(box) == z:
            return i
    return -1


def to_str(state):
    boxes = boxes_from_state(state)
    return str([f"p={pos(box)}, s={size(box)}, z={zone(box)}" for box in boxes]) + f", t={state[-1]}"


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


def in_front(box1, box2):
    """ Returns True if box1 is in front of box2, False otherwise. """
    return pos(box1)[REMOVE_DIR] > pos(box2)[REMOVE_DIR]


def null_box():
    """ Creates a box that is not valid for placeholding."""
    return make(np.zeros(DIM), np.zeros(DIM), -1)

def top_and_bottom(box):
    """ Computes the top and bottom points of the box. """
    bottom_points = set()
    top_points = set()
    p = pos(box)
    s = size(box)
    for i in range(s[0]+1):
        for j in range(s[1]+1):
            bottom_points.add(tuple(p + np.array([i, j, 0])))
            top_points.add(tuple(p + np.array([i, j, s[2]])))
    return top_points, bottom_points
