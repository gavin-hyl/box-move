import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import our environment and constants.
from BoxMoveEnvGym import BoxMoveEnvGym
from Constants import ZONE0, ZONE1
from QNet import CNNQNetwork

def pad_to_shape(array, target_shape):
    """
    Pads the input 3D array with zeros at the end of each axis so that its shape matches target_shape.
    """
    pad_width = [(0, t - s) for s, t in zip(array.shape, target_shape)]
    return np.pad(array, pad_width, mode='constant', constant_values=0)

def generate_training_data(num_episodes=50, max_steps=20):
    """
    Generate training samples by running the BoxMove environment.
    
    For each step, we extract a 3D state and action representation.
    The state is composed of two channels (ZONE0 and ZONE1) from state_3d(),
    and the action is composed of two channels (ZONE0 and ZONE1) from action_3d().
    
    Since ZONE0 and ZONE1 have different spatial dimensions, we pad both to a common shape.
    The common shape is computed as the elementwise maximum of ZONE0 and ZONE1.
    
    Returns:
        data: List of tuples (state_np, action_np, reward)
              where state_np has shape [2, D, H, W] and action_np has shape [2, D, H, W]
    """
    data = []
    env = BoxMoveEnvGym(horizon=50, n_boxes=15)
    
    # Determine common shape: elementwise maximum between ZONE0 and ZONE1.
    target_shape = (max(ZONE0[0], ZONE1[0]),
                    max(ZONE0[1], ZONE1[1]),
                    max(ZONE0[2], ZONE1[2]))
    
    for ep in range(num_episodes):
        # Reset the environment.
        env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Get the current 3D state representation.
            # state_3d() returns a list: [zone0_dense, zone1_dense]
            state_3d = env.env.state_3d()  # using the underlying environment
            # Pad each zone to the common target shape.
            padded_state = [pad_to_shape(zone, target_shape) for zone in state_3d]
            # Stack both zones along a new channel dimension.
            state_np = np.stack(padded_state, axis=0)
            
            # Retrieve the list of valid actions.
            valid_actions = env.env.actions()
            if len(valid_actions) == 0:
                break
            
            # Randomly choose one valid action.
            chosen_action = np.random.choice(valid_actions)
            
            # Get the 3D representation for the action.
            # action_3d() returns a list: [zone0_dense, zone1_dense]
            action_3d = env.env.action_3d(chosen_action)
            padded_action = [pad_to_shape(zone, target_shape) for zone in action_3d]
            # Stack both zones along a new channel dimension.
            action_np = np.stack(padded_action, axis=0)
            
            # Find the discrete action index corresponding to chosen_action.
            action_idx = None
            for idx, act in env._action_map.items():
                if act == chosen_action:
                    action_idx = idx
                    break
            if action_idx is None:
                steps += 1
                continue  # Skip if we can't determine the discrete index.
            
            # Take the step in the environment.
            next_state, reward, done, truncated, info = env.step(action_idx)
            
            # Append the training sample.
            data.append((state_np.copy(), action_np.copy(), reward))
            steps += 1
    return data

def main():
    # Generate training data.
    print("Generating training data...")
    data = generate_training_data(num_episodes=100, max_steps=20)
    print(f"Collected {len(data)} samples.")

    # Convert the list of samples into torch tensors.
    # The network expects inputs of shape [batch_size, channels, depth, height, width].
    states = torch.tensor(np.array([sample[0] for sample in data]), dtype=torch.float32)
    actions = torch.tensor(np.array([sample[1] for sample in data]), dtype=torch.float32)
    rewards = torch.tensor(np.array([sample[2] for sample in data]), dtype=torch.float32).unsqueeze(1)

    # Create a dataset and a DataLoader.
    dataset = TensorDataset(states, actions, rewards)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Instantiate the CNN Q network.
    # Note: The network now expects state inputs with spatial dimensions equal to target_shape.
    net = CNNQNetwork(state_channels=2, action_channels=2)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    num_epochs = 30
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for state_batch, action_batch, reward_batch in loader:
            optimizer.zero_grad()
            q_pred = net(state_batch, action_batch)
            loss = loss_fn(q_pred, reward_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * state_batch.size(0)
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Save the trained model.
    torch.save(net.state_dict(), "cnn_q_network.pth")
    print("Training complete. Model saved as cnn_q_network.pth.")

if __name__ == "__main__":
    main()
