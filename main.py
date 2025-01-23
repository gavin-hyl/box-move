from BoxEnv import Box, BoxMoveEnvironment, setup
import numpy as np

setup(remove_ax='z')

# state = np.array([1, 1, 0, 0])
# box = Box.make(np.array([0, 0, 0]), np.array([1, 1, 1]), 0)
# box2 = Box.make(np.array([1, 1, 1]), np.array([1, 1, 1]), 0)
# state = Box.state_from_boxes([box, box2], 0)

# bme = BoxMoveEnvironment([[2, 3, 3], [1, 1, 1]])

# actions = bme.actions(state)
# for a in actions:
#     print(a)
# state, r, done = bme.step(state, actions[0])
# state = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0])

# print(Box.to_str(state))
# print(bme.is_valid(state))

# actions = bme.actions(state)


box = Box.make(np.array([0, 0, 0]), np.array([1, 1, 1]), 0)
box2 = Box.make(np.array([1, 0, 0]), np.array([1, 1, 1]), 1)
# box3 = Box.make(np.array([2, 0, 0]), np.array([1, 3, 3]), 0)

boxes = [box, box2]
# boxes = [box, box2, box3]
state = Box.state_from_boxes(boxes)
bme = BoxMoveEnvironment([[3, 2, 2], [2, 1, 1]])
# print(Box.top_and_bottom(box))
print(bme._zone_top_bottom(state, 0))
# print(bme.support(state, 0))

actions = bme.actions(state)
for a in actions:  # missing actions relating to box
    print(a)