import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import our environment and constants.
from BoxMoveEnvGym import BoxMoveEnvGym
from Constants import ZONE0, ZONE1, MODEL_DIR, MODEL_NAME, DATA_DIR
from QNet import CNNQNetwork

def generate_training_data(num_episodes=50, max_steps=20):
    """
    Generate training samples by running the BoxMove environment.
    
    For each step, we extract a 3D state and action representation.
    The state is represented as two arrays (one for each zone) obtained from state_3d(),
    and the action is represented as two arrays from the chosen action.
    
    Returns:
        data: List of tuples (state_zone0, state_zone1, action_zone0, action_zone1, reward)
    """
    data = []
    env = BoxMoveEnvGym(horizon=50, n_boxes=15)
    
    for ep in range(num_episodes):
        env.reset()
        done = False
        steps = 0
        episode_data = []
        episode_reward = 0
        
        while not done and steps < max_steps:
            # Get current 3D state (list: [zone0_dense, zone1_dense])
            state_3d = env.env.state_3d()  # using the underlying environment
            
            # Retrieve valid actions.
            valid_actions = env.env.actions()
            if len(valid_actions) == 0:
                break
            
            # Randomly choose one valid action.
            chosen_action = np.random.choice(valid_actions)
            
            # Get the 3D representation for the action.
            action_3d = env.env.action_3d(chosen_action)
            
            # Find the discrete action index corresponding to chosen_action.
            action_idx = None
            for idx, act in env._action_map.items():
                if act == chosen_action:
                    action_idx = idx
                    break
            if action_idx is None:
                steps += 1
                continue
            
            # Take the step.
            next_state, reward, done, truncated, info = env.step(action_idx)
            
            # Append the training sample:
            # (state_zone0, state_zone1, action_zone0, action_zone1, reward)
            episode_data.append((state_3d[0].copy(), state_3d[1].copy(),
                                 action_3d[0].copy(), action_3d[1].copy(), reward))
            # episode_reward = reward
            steps += 1
        
        for d in episode_data:
            data.append((d[0], d[1], d[2], d[3], d[4]))
    
    # Save training data.
    return data

def main():
    print("Generating training data...")
    data = generate_training_data(num_episodes=100, max_steps=20)
    print(f"Collected {len(data)} samples.")
    
    # Convert samples into torch tensors.
    # For each zone, add a channel dimension to get shape [batch_size, 1, D, H, W].
    states_zone0 = torch.tensor(np.array([sample[0] for sample in data]), dtype=torch.float32).unsqueeze(1)
    states_zone1 = torch.tensor(np.array([sample[1] for sample in data]), dtype=torch.float32).unsqueeze(1)
    actions_zone0 = torch.tensor(np.array([sample[2] for sample in data]), dtype=torch.float32).unsqueeze(1)
    actions_zone1 = torch.tensor(np.array([sample[3] for sample in data]), dtype=torch.float32).unsqueeze(1)
    rewards = torch.tensor(np.array([sample[4] for sample in data]), dtype=torch.float32).unsqueeze(1)
    
    # Create a dataset with separate inputs for each zone.
    dataset = TensorDataset(states_zone0, states_zone1, actions_zone0, actions_zone1, rewards)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Instantiate the CNN Q-network.
    net = CNNQNetwork()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    num_epochs = 30
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for state_z0_batch, state_z1_batch, action_z0_batch, action_z1_batch, reward_batch in loader:
            optimizer.zero_grad()
            # Use the network's forward_separate method.
            q_pred = net.forward_separate(state_z0_batch, state_z1_batch,
                                          action_z0_batch, action_z1_batch)
            loss = loss_fn(q_pred, reward_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * state_z0_batch.size(0)
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    model_path = f"{MODEL_DIR}/{MODEL_NAME}"
    torch.save(net.state_dict(), model_path)
    print(f"Training complete. Model saved as {model_path}")

if __name__ == "__main__":
    main()
