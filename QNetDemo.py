import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from BoxMoveEnvGym import BoxMoveEnvGym
from QNet import CNNQNetwork
from Constants import MODEL_DIR, MODEL_NAME

plt.ion()

def get_discrete_action_index(env, chosen_action):
    """Helper to get the discrete action index for a given BoxAction."""
    for idx, act in env._action_map.items():
        if act == chosen_action:
            return idx
    return None

def main():
    # Initialize the environment and load the pretrained CNN.
    env = BoxMoveEnvGym(horizon=50, n_boxes=10)
    net = CNNQNetwork()
    net.load_state_dict(torch.load(f"{MODEL_DIR}/{MODEL_NAME}"))
    net.eval()  # Set network to evaluation mode.
    
    done = False
    step = 0

    while not done:
        # Get the current state (without padding)
        # state_3d() returns [zone0_dense, zone1_dense] with shapes ZONE0 and ZONE1.
        state_3d = env.env.state_3d()
        # Convert to torch tensors and add batch and channel dimensions.
        state_zone0 = torch.tensor(state_3d[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        state_zone1 = torch.tensor(state_3d[1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Retrieve valid actions.
        valid_actions = env.env.actions()
        if len(valid_actions) == 0:
            print("No valid actions remain.")
            break

        # Loop over valid actions and compute Q-values.
        best_q = -float('inf')
        best_action = None
        for action in valid_actions:
            # Get the 3D action representation (returns [zone0_dense, zone1_dense])
            action_3d = env.env.action_3d(action)
            action_zone0 = torch.tensor(action_3d[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            action_zone1 = torch.tensor(action_3d[1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                q_value = net.forward(state_zone0, state_zone1, action_zone0, action_zone1)
            
            if q_value.item() > best_q:
                best_q = q_value.item()
                best_action = action
        
        print(f"Step {step}: Best Q value = {best_q:.4f} for action {best_action}")
        
        # Visualize the current scene.
        env.render()
        fig = plt.gcf()
        fig.suptitle(f"Step {step} | Best Q value: {best_q:.4f}", fontsize=16)
        plt.draw()
        plt.pause(2)
        plt.close('all')
        
        # Determine the discrete action index for the best action.
        discrete_action = get_discrete_action_index(env, best_action)
        if discrete_action is None:
            print("Could not determine discrete action index; skipping step.")
            continue
        
        # Take the chosen (optimal) action.
        _, reward, done, truncated, info = env.step(discrete_action)
        print(f"Reward: {reward:.4f}")
        
        step += 1
        time.sleep(0.5)

    print("Episode finished.")

if __name__ == "__main__":
    main()
