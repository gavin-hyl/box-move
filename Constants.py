import numpy as np
DIM = 3
STATE_DIM = 2*DIM + 1
REMOVE_DIR = 2  # 0, 1, 2 for x, y, z
ZONE0 = (3, 4, 5)
ZONE1 = (3, 3, 3)
ZONE_SIZES = [ZONE0, ZONE1]
ZONE0_DENSE = np.zeros(ZONE0, dtype=int)
ZONE1_DENSE = np.zeros(ZONE1, dtype=int)
DIRECTIONS = ['x', 'y', 'z']

# print("===================================================")
# print("Constants:")
# print(f" * Space Dimension = {DIM}")
# print(f" * Box State Dimension = {STATE_DIM}")
# print(f" * Removal Direction = {DIRECTIONS[REMOVE_DIR]}")
# print(f" * Zone Sizes = {ZONE_SIZES}")
# print("===================================================")