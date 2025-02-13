import numpy as np
from Constants import GEO_DIM, BOX_DIM, REMOVE_DIR


def pos(box, p=None):
    if len(box) != BOX_DIM:
        print(f"Box: {box}")
        raise ValueError(f"Box must be of length {BOX_DIM}, not {len(box)}")
    if p is not None:
        box[:GEO_DIM] = p
    return box[:GEO_DIM]


def size(box, size=None):
    if size is not None:
        box[GEO_DIM : 2 * GEO_DIM] = size
    return box[GEO_DIM : 2 * GEO_DIM]


def zone(box, zone=None):
    if zone is not None:
        box[2 * GEO_DIM] = zone
    return box[2 * GEO_DIM]


def make(p, s, zone):
    rep = np.zeros(BOX_DIM, dtype=int)
    for i, x in enumerate(p):
        rep[i] = x
    for i, x in enumerate(s):
        rep[i + GEO_DIM] = x
    rep[2 * GEO_DIM] = zone
    return rep


def access_from_state(state, idx, new_box=None):
    """Returns the box at the given index in the state vector."""
    idx *= BOX_DIM
    if idx < 0 or idx >= len(state) - 1:
        raise ValueError(f"Index {idx} out of bounds for state of length {len(state)}")
    if new_box is not None:
        state[idx : idx + BOX_DIM] = new_box
    return state[idx : idx + BOX_DIM]


def boxes_from_state(state):
    """Extracts a list of boxes from a state vector."""
    boxes = []
    for i in range(0, len(state) - 1, BOX_DIM):
        boxes.append(state[i : i + BOX_DIM])
    return boxes


def state_from_boxes(boxes, t=0):
    """Creates a state vector from a list of boxes and a time step."""
    state = np.zeros(len(boxes) * (BOX_DIM) + 1, dtype=int)
    for i, box in enumerate(boxes):
        access_from_state(state, i, box)
    state[-1] = t
    return state


def box_idx(state, p, z):
    """Returns the index of the box with the given position and zone."""
    boxes = boxes_from_state(state)
    for i, box in enumerate(boxes):
        if (pos(box) == p).all() and zone(box) == z:
            return i
    return -1


def to_str(state):
    boxes = boxes_from_state(state)
    return (
        str([f"p={pos(box)}, s={size(box)}, z={zone(box)}" for box in boxes])
        + f", t={state[-1]}"
    )


def key_from_pair(box1, box2):
    """Returns a hashable key for a pair of boxes."""
    box1_first = False
    for i in range(BOX_DIM):
        box1_first = box1[i] < box2[i]
        if box1[i] != box2[i]:
            break
    it = iter([box1, box2]) if box1_first else iter([box2, box1])
    key = []
    for box in it:
        key.extend(list(box))
    return tuple(key)


def in_front(box1, box2):
    """Returns True if box1 is in front of box2, False otherwise."""
    return pos(box1)[REMOVE_DIR] > pos(box2)[REMOVE_DIR]


def null_box():
    """Creates a box that is not valid for placeholding."""
    return make(np.zeros(GEO_DIM), np.zeros(GEO_DIM), -1)


def is_null(box):
    """Checks if the box is null."""
    return (box == null_box()).all()


def top_and_bottom(box):
    """Computes the top and bottom points of the box."""
    bottom_points = set()
    top_points = set()
    p = pos(box)
    s = size(box)
    for i in range(s[0] + 1):
        for j in range(s[1] + 1):
            bottom_points.add(tuple(p + np.array([i, j, 0])))
            top_points.add(tuple(p + np.array([i, j, s[2]])))
    return top_points, bottom_points


def top_face(box):
    """ Returns a set of points representing the top face of the box. Each tile is represented by its bottom-left corner. """
    p, s = pos(box), size(box)
    points = []
    for i in range(s[0]):
        for j in range(s[1]):
            points.append(p + np.array([i, j, s[2]]))
    points = [tuple(point) for point in points]
    return set(points)

def bottom_face(box):
    """ Returns a set of points representing the bottom face of the box. Each tile is represented by its bottom-left corner. """
    p, s = pos(box), size(box)
    points = []
    for i in range(s[0]):
        for j in range(s[1]):
            points.append(p + np.array([i, j, 0]))
    points = [tuple(point) for point in points]
    return set(points)