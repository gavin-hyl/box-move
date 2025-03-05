import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm

# Import our environment and constants.
from Constants import MODEL_DIR
from QNet import CNNQNetwork
from Utils import LossTracker
from DataHandling import load_data


def main():
    MODEL_NAME = "cnn_qnet"

    print("Generating training data...")
    data = load_data("training_data300")
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
    
    # Split the dataset into training and validation sets (80% training, 20% validation).
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Instantiate the CNN Q-network.
    net = CNNQNetwork()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    num_epochs = 50
    print("Starting training...")
    tracker = LossTracker(patience=10)
    
    for epoch in tqdm(range(num_epochs), desc="Epochs", unit="epoch"):
        # Training phase
        net.train()
        epoch_train_loss = 0.0
        for state_z0_batch, state_z1_batch, action_z0_batch, action_z1_batch, reward_batch in tqdm(train_loader, desc="Train batches", leave=False):
            optimizer.zero_grad()
            q_pred = net(state_z0_batch, state_z1_batch, action_z0_batch, action_z1_batch)
            loss = loss_fn(q_pred, reward_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * state_z0_batch.size(0)
        epoch_train_loss /= train_size
        
        # Validation phase
        net.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for state_z0_batch, state_z1_batch, action_z0_batch, action_z1_batch, reward_batch in tqdm(val_loader, desc="Val batches", leave=False):
                q_pred = net(state_z0_batch, state_z1_batch, action_z0_batch, action_z1_batch)
                loss = loss_fn(q_pred, reward_batch)
                epoch_val_loss += loss.item() * state_z0_batch.size(0)
        epoch_val_loss /= val_size
        
        tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        
        # Early stopping check
        tracker(epoch_train_loss, epoch_val_loss)
        if tracker.early_stop:
            tqdm.write("Early stopping triggered.")
            break
        
        if epoch % 5 == 0:
            model_path = f"{MODEL_DIR}/{MODEL_NAME}_epoch{epoch}.pth"
            torch.save(net.state_dict(), model_path)

    model_path = f"{MODEL_DIR}/{MODEL_NAME}_epoch{epoch}.pth"
    tqdm.write(f"Training complete. Model saved as {model_path}")
    tracker.render()


if __name__ == "__main__":
    main()
