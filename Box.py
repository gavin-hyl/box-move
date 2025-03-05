import numpy as np
from Constants import GEOM_DIM, BOX_DIM, VAL_IDX, ZONE_IDX


class Box:
    def __init__(self, pos=None, size=None, zone=None, val=None):
        self.pos = pos if pos is not None else np.zeros(GEOM_DIM)
        self.size = size if size is not None else np.zeros(GEOM_DIM)
        self.zone = zone if zone is not None else -1
        self.val = val if val is not None else 0
    
    def __eq__(self, other):
        return self.pos == other.pos and self.size == other.size and self.zone == other.zone and self.val == other.val
    
    def __str__(self):
        return f"pos={self.pos}, size={self.size}, zone={self.zone}, val={self.val}"
    
    def __repr__(self):
        return str(self)

    def top_face(self):
        """ Returns a set of points representing the top face of the box. Each tile is represented by its bottom-left corner. """
        points = []
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                points.append(self.pos + np.array([i, j, self.size[2]]))
        points = [tuple(point) for point in points]
        return set(points)

    def bottom_face(self):
        """ Returns a set of points representing the bottom face of the box. Each tile is represented by its bottom-left corner. """
        points = []
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                points.append(self.pos + np.array([i, j, 0]))
        points = [tuple(point) for point in points]
        return set(points)
    



def pos(box, p=None):
    if len(box) != BOX_DIM:
        print(f"Box: {box}")
        raise ValueError(f"Box must be of length {BOX_DIM}, not {len(box)}")
    if p is not None:
        box[:GEOM_DIM] = p
    return box[:GEOM_DIM]


def size(box, size=None):
    if size is not None:
        box[GEOM_DIM : 2 * GEOM_DIM] = size
    return box[GEOM_DIM : 2 * GEOM_DIM]


def zone(box, zone=None):
    if zone is not None:
        box[ZONE_IDX] = zone
    return box[ZONE_IDX]


def val(box, val=None):
    if val is not None:
        box[VAL_IDX] = val
    return box[VAL_IDX]


def make(p, s, zone, val):
    rep = np.zeros(BOX_DIM, dtype=int)
    for i, x in enumerate(p):
        rep[i] = x
    for i, x in enumerate(s):
        rep[i + GEOM_DIM] = x
    rep[ZONE_IDX] = zone
    rep[VAL_IDX] = val
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
        str([f"p={pos(box)}, s={size(box)}, z={zone(box)}, v={val(box)}" for box in boxes])
        + f", t={state[-1]}"
    )


def null_box():
    """Creates a box that is not valid for placeholding."""
    return make(np.zeros(GEOM_DIM), np.zeros(GEOM_DIM), -1)


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