import numpy as np

from BoxAction import BoxAction
import Core
import Box
from Constants import ZONE_SIZES

def dense_zone(zone: int):
    return np.zeros(ZONE_SIZES[zone], dtype=int)

def dense_state(state: np.ndarray):
    """ When printed, the outermost layer represents x, down represents y, and right represents z. """
    representations = [dense_zone(i) for i in [0, 1]]
    for idx, box in enumerate(Box.boxes_from_state(state)):
        p, s, zone = Box.pos(box), Box.size(box), Box.zone(box)
        for x in range(p[0], p[0] + s[0]):
            for y in range(p[1], p[1] + s[1]):
                for z in range(p[2], p[2] + s[2]):
                    if representations[zone][x, y, z] != 0:
                        raise ValueError(f"Overlapping boxes at position {x, y, z}")
                    representations[zone][x, y, z] = idx + 1
    return representations

def dense_action(state, action: BoxAction):
    rep_init = dense_state(state)
    rep_final = dense_state(Core.transition(state, action))
    return [r1 - r0 for r0, r1 in zip(rep_init, rep_final)]


if __name__ == '__main__':
    state = Box.state_from_boxes([
        Box.make([0, 0, 0], [1, 1, 3], 0),
        Box.make([1, 0, 0], [1, 1, 1], 0),
        Box.make([0, 1, 0], [1, 1, 1], 0),
    ])
    print("Initial State:")
    print(dense_state(state)[0])
    actions = Core.actions(state)
    print("Sample Action:")
    for rep in dense_action(state, actions[0]):
        print(rep)