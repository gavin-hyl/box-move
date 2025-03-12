#==========================================================
#            Constants for Box Move Simulation
#==========================================================

import numpy as np

GEOM_DIM = 3
ZONE_IDX = 2*GEOM_DIM
VAL_IDX = ZONE_IDX + 1
BOX_DIM = VAL_IDX + 1

REMOVE_DIR = 2  # 0, 1, 2 for x, y, z
# Modified zone dimensions to make the problem more challenging
# Original: ZONE0 = (3, 4, 5) and ZONE1 = (4, 3, 3)
ZONE0 = (4, 5, 5)  # Larger source zone - more boxes and more options
ZONE1 = (3, 3, 4)  # Smaller target zone - more careful packing needed
ZONE_SIZES = [ZONE0, ZONE1]
def zone0_dense_cpy():
    return np.zeros(ZONE0, dtype=int)

def zone1_dense_cpy():
    return np.zeros(ZONE1, dtype=int)

ZONE0_DENSE = np.zeros(ZONE0, dtype=int)
ZONE1_DENSE = np.zeros(ZONE1, dtype=int)
DIRECTIONS = ['x', 'y', 'z']


#==========================================================
#                Constants for Training
#==========================================================
MODEL_DIR = "models"
DATA_DIR = "data"

# Rewards for different actions
MOVE_REWARD = 0.1       # Small reward for any valid move
SUCCESS_REWARD = 1.0    # Reward for successfully moving a box to zone 1
COMPLETION_REWARD = 5.0 # Reward for optimal placement of all boxes

# print("===================================================")
# print("Constants:")
# print(f" * Space Dimension = {DIM}")
# print(f" * Box State Dimension = {STATE_DIM}")
# print(f" * Removal Direction = {DIRECTIONS[REMOVE_DIR]}")
# print(f" * Zone Sizes = {ZONE_SIZES}")
# print("===================================================")