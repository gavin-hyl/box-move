import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from BoxMoveEnvGym import BoxMoveEnvGym
from QNet import CNNQNetwork
from Constants import ZONE0, ZONE1

plt.ion()

def pad_to_shape(array, target_shape):
    """
    Pads the input 3D array with zeros at the end of each axis so that its shape matches target_shape.
    """
    pad_width = [(0, t - s) for s, t in zip(array.shape, target_shape)]
    return np.pad(array, pad_width, mode='constant', constant_values=0)

def get_common_shape(zone0_shape, zone1_shape):
    """
    Computes a target shape as the elementwise maximum of two zone shapes.
    """
    return (max(zone0_shape[0], zone1_shape[0]),
            max(zone0_shape[1], zone1_shape[1]),
            max(zone0_shape[2], zone1_shape[2]))

def get_discrete_action_index(env, chosen_action):
    """Helper to get the discrete action index for a given BoxAction."""
    for idx, act in env._action_map.items():
        if act == chosen_action:
            return idx
    return None

def main():
    # Initialize the environment and the CNN.
    env = BoxMoveEnvGym(horizon=50, n_boxes=10)
    net = CNNQNetwork(state_channels=2, action_channels=2)
    # Optionally load a pretrained model:
    net.load_state_dict(torch.load("cnn_q_network.pth"))
    net.eval()  # Set to evaluation mode.
    
    # Reset the environment.
    env.reset()
    
    # Determine the common target shape.
    target_shape = get_common_shape(ZONE0, ZONE1)  # For example, if ZONE0=(5,4,3) and ZONE1=(3,5,4), then target_shape=(5,5,4)
    
    done = False
    step = 0

    while not done:
        # Get the current state representation as a 3D pair for zone0 and zone1.
        # state_3d() returns a list [zone0_dense, zone1_dense] with shapes ZONE0 and ZONE1 respectively.
        state_3d = env.env.state_3d()
        # Pad each zone to the common target shape.
        padded_state = [pad_to_shape(zone, target_shape) for zone in state_3d]
        # Stack along a new channel dimension: result shape [2, D, H, W].
        state_np = np.stack(padded_state, axis=0)
        
        # Choose a valid action at random.
        valid_actions = env.env.actions()
        if len(valid_actions) == 0:
            print("No valid actions remain.")
            break
        chosen_action = np.random.choice(valid_actions)
        
        # Get the 3D representation for the action.
        # action_3d() returns a list [zone0_dense, zone1_dense].
        action_3d = env.env.action_3d(chosen_action)
        padded_action = [pad_to_shape(zone, target_shape) for zone in action_3d]
        action_np = np.stack(padded_action, axis=0)
        
        # Convert the state and action representations to torch tensors.
        # Add a batch dimension so that the shape becomes [1, channels, D, H, W].
        state_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor(action_np, dtype=torch.float32).unsqueeze(0)
        
        # Compute the Q value from the CNN.
        with torch.no_grad():
            q_value = net(state_tensor, action_tensor).item()
        
        print(f"Step {step}: Q value = {q_value:.4f}")
        
        # Visualize the current scene.
        env.render()  # Calls env.env.visualize_scene() internally.
        fig = plt.gcf()
        fig.suptitle(f"Step {step} | Q value: {q_value:.4f}", fontsize=16)
        plt.draw()
        plt.pause(2)  # Pause for 2 seconds.
        plt.close('all')
        
        # Determine the discrete action index corresponding to the chosen action.
        discrete_action = get_discrete_action_index(env, chosen_action)
        if discrete_action is None:
            print("Could not determine discrete action index; skipping step.")
            continue
        
        # Take the step in the environment.
        _, reward, done, truncated, info = env.step(discrete_action)
        print(f"Reward: {reward:.4f}")
        
        step += 1
        time.sleep(0.5)  # Optional pause between steps.

    print("Episode finished.")

if __name__ == "__main__":
    main()
