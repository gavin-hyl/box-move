# Box-Moving Environment


## Explanation
- This environment is designed following a (deterministic) MDP structure. The task is to move boxes from one zone to another. 
- Given $n$ boxes, the vector representation for each box is $7$ long: $(p, s, z)$ where $p$ is the position, $s$ is the size (also a $3$-vector), and $z$ is zone the box is currently in. 
- The state vector combines all boxes in no particular order. It also records the timestep of the current state, which gives a $7n+1$ long state vector. 
- The action set at every step includes all actions that take accessible boxes (those that are unobstructed in the `remove_ax` direction) to an accessible location (unobstructed in `remove_ax`, and all points on the bottom are supported by other boxes). Note that this includes actions from the same zone to itself. The actions are objects with attributes $(p_i, p_f)$ and comparison methods.
- The reward is given by the occupancy of the target zone, which is computed as the ratio of volume of boxes to the volume of the zone

## BoxMove Environment & CNN Q-Network

This repository contains a custom reinforcement learning environment called **BoxMove** along with a convolutional Q-network implementation for evaluating state–action pairs. The environment simulates the movement of boxes between two zones with differing spatial dimensions. It also includes visualization tools that draw both the bounding boxes of the zones and the bounding boxes (wireframes) of the individual boxes.

---

## Overview

The main components of this project include:

- **Custom Environment:**  
  The **BoxMove** environment simulates moving boxes from **ZONE0** to **ZONE1**. The zones have different sizes (defined in [Constants.py](Constants.py)), and the environment provides 3D state representations and actions that move boxes between zones.

- **Q-Network:**  
  A convolutional neural network (CNN) implemented in [QNet.py](QNet.py) evaluates state–action pairs to compute Q-values. It includes separate convolutional branches for each zone, enabling it to accommodate the different dimensions of the zones. An extended method (`forward_separate`) has been added to process the zones without any padding.

- **Training Pipeline:**  
  The training script ([QNetTrain.py](QNetTrain.py)) generates training data by running the environment and then trains the CNN Q-network using PyTorch.

- **Demo & Testing:**  
  - [QNetDemo.py](QNetDemo.py) demonstrates how to load a pretrained model and use it to select the optimal action based on Q-values computed for all valid actions.  
  - [BoxMoveTest.py](BoxMoveTest.py) provides tests for the environment and its gym wrapper.

- **Gym Wrapper:**  
  [BoxMoveEnvGym.py](BoxMoveEnvGym.py) wraps the BoxMove environment into a [Gym](https://gym.openai.com/) environment, enabling integration with other reinforcement learning frameworks.

- **Visualization:**  
  The environment’s visualization function (in [BoxMoveEnv.py](BoxMoveEnv.py)) renders a 3D scene showing:
  - The overall zone boundaries (bounding boxes of the zones).
  - Each box rendered as a semi-transparent cuboid.
  - The bounding box (wireframe) of each box drawn in a zone-specific color (blue for zone 0 and red for zone 1).

---

## Repository Structure

- **QNet.py**  
  Defines the CNN Q-network with separate convolutional branches for the two zones. Also includes a `forward_separate` method for processing raw zone representations.

- **QNetTrain.py**  
  Generates training data from the BoxMove environment and trains the Q-network.

- **QNetDemo.py**  
  Loads a pretrained model and demonstrates optimal action selection by computing Q-values for all valid actions. It also visualizes the environment with updated bounding boxes.

- **BoxMoveEnv.py**  
  Implements the BoxMove environment including state generation, action handling, reward computation, and visualization.

- **BoxMoveEnvGym.py**  
  Provides a Gym-compatible wrapper for the BoxMove environment.

- **BoxMoveTest.py**  
  Contains tests for the BoxMove environment and its Gym wrapper.

- **BoxAction.py**  
  Defines the `BoxAction` class for encoding movements of boxes between zones.

- **Constants.py**  
  Stores constants used throughout the project (e.g., dimensions for ZONE0 and ZONE1).

- **Box.py**  
  Provides helper functions for creating, manipulating, and representing boxes within the environment.

---

## Setup & Dependencies
Install the required dependencies using pip:

```bash
pip install torch numpy matplotlib gym
```