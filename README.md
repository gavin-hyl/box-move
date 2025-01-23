# Box-Moving Environment


## Explanation
- This environment is designed following a (deterministic) MDP structure. The task is to move boxes from one zone to another. 
- Given $n$ boxes, the vector representation for each box is $7$ long: $(p, s, z)$ where $p$ is the position, $s$ is the size (also a $3$-vector), and $z$ is zone the box is currently in. 
- The state vector combines all boxes in no particular order. It also records the timestep of the current state, which gives a $7n+1$ long state vector. 
- The action set at every step includes all actions that take accessible boxes (those that are unobstructed in the `remove_ax` direction) to an accessible location (unobstructed in `remove_ax`, and all points on the bottom are supported by other boxes). Note that this includes actions from the same zone to itself. The actions are objects with attributes $(p_i, z_i, p_f, z_f)$ (should convert to np arrays!)
- The reward is given by the occupancy of the target zone, which is computed as the ratio of volume of boxes to the volume of the zone


## Usage
```py
box = Box.make(np.array([0, 0, 0]), np.array([1, 1, 1]), 0) # make a box with pos, size, zone
state = Box.state_from_boxes([box, box2], 0)                # create a state vector with a list of boxes and the current time
bme = BoxMoveEnvironment([[2, 3, 3], [1, 1, 1]])            # create an environment with two zones with the respective sizes
actions = bme.actions(state)                                # compute all valid actions
state, r, done = bme.step(state, actions[0])                # take an action
````