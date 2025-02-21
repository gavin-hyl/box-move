import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import our environment and constants.
from BoxMoveEnvGym import BoxMoveEnvGym
from Constants import ZONE0, ZONE1
from QNetwork import CNNQNetwork

def generate_training_data(num_episodes=50, max_steps=20):
    """
    Generate training samples by running the BoxMove environment.
    
    For each step, we extract a 3D state and action representation.
    The state is taken from zone0_dense (from state_3d) and the action
    from zone1_dense (from action_3d). The reward returned after executing
    the action is used as the target Q value.
    
    Returns:
        data: List of tuples (state_np, action_np, reward)
    """
    data = []
    env = BoxMoveEnvGym(horizon=50, n_boxes=5)
    
    for ep in range(num_episodes):
        # Reset the environment.
        env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Get the current 3D state representation.
            # state_3d() returns a list: [zone0_dense, zone1_dense].
            state_3d = env.env.state_3d()  # using the underlying environment
            # We'll use the zone0 representation as the state input.
            state_np = state_3d[0]
            
            # Retrieve the list of valid actions.
            valid_actions = env.env.actions()
            if len(valid_actions) == 0:
                break
            
            # Randomly choose one valid action.
            chosen_action = np.random.choice(valid_actions)
            # For compatibility with action_3d(), set zone_from to 0 if not already present.
            if not hasattr(chosen_action, 'zone_from'):
                setattr(chosen_action, 'zone_from', 0)
            
            # Get the 3D representation for the action.
            # action_3d() returns a list: [zone0_dense, zone1_dense].
            action_3d = env.env.action_3d(chosen_action)
            # We use the zone1 representation as the action input.
            action_np = action_3d[1]
            
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
            # We use copies to ensure the arrays are not modified later.
            data.append((state_np.copy(), action_np.copy(), reward))
            steps += 1
    return data

def main():
    # Generate training data.
    print("Generating training data...")
    data = generate_training_data(num_episodes=50, max_steps=20)
    print(f"Collected {len(data)} samples.")

    # Convert the list of samples into torch tensors.
    # Our network expects inputs of shape [batch_size, channels, depth, height, width].
    # We'll add a channel dimension (value 1) to each sample.
    states = torch.tensor(np.array([sample[0] for sample in data]), dtype=torch.float32).unsqueeze(1)
    actions = torch.tensor(np.array([sample[1] for sample in data]), dtype=torch.float32).unsqueeze(1)
    rewards = torch.tensor(np.array([sample[2] for sample in data]), dtype=torch.float32).unsqueeze(1)

    # Create a dataset and a DataLoader.
    dataset = TensorDataset(states, actions, rewards)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Instantiate the CNN Q network.
    net = CNNQNetwork(state_channels=1, action_channels=1)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    num_epochs = 10
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for state_batch, action_batch, reward_batch in loader:
            optimizer.zero_grad()
            # Forward pass: predict Q values.
            q_pred = net(state_batch, action_batch)
            loss = loss_fn(q_pred, reward_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * state_batch.size(0)
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Optionally, save the trained model.
    torch.save(net.state_dict(), "cnn_q_network.pth")
    print("Training complete. Model saved as cnn_q_network.pth.")

if __name__ == "__main__":
    main()
