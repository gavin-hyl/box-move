import subprocess
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm

from QNetEnsemble import QNetEnsemble
from Constants import MODEL_DIR
from DataHandling import load_data

def get_latest_model_path(folder_name, model_name):
    """
    Searches MODEL_DIR/folder_name for the latest model file matching the pattern
    {model_name}_epoch*.pth and returns its path.
    """
    pattern = os.path.join(MODEL_DIR, folder_name, f"{model_name}_epoch*.pth")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No model files found for {model_name} in {MODEL_DIR}/{folder_name}")
    
    def get_epoch(file_path):
        basename = os.path.basename(file_path)
        try:
            epoch_str = basename.split("_epoch")[1].split(".pth")[0]
            return int(epoch_str)
        except (IndexError, ValueError):
            return -1

    latest_file = max(files, key=get_epoch)
    return latest_file

def train_ensemble_logits(ensemble_model, train_loader, val_loader, num_epochs, device):
    """
    Fine-tune the ensemble logits while freezing the individual network parameters.
    """
    # Freeze individual networks
    for net in ensemble_model.networks:
        for param in net.parameters():
            param.requires_grad = False

    # Only ensemble_logits will be updated.
    optimizer = optim.Adam([ensemble_model.ensemble_logits], lr=0.001)
    loss_fn = nn.SmoothL1Loss()

    best_val_loss = float('inf')
    best_logits = None

    for epoch in tqdm(range(num_epochs), desc="Training ensemble logits"):
        ensemble_model.train()
        train_loss = 0.0
        for state_z0, state_z1, action_z0, action_z1, reward in train_loader:
            optimizer.zero_grad()
            state_z0 = state_z0.to(device)
            state_z1 = state_z1.to(device)
            action_z0 = action_z0.to(device)
            action_z1 = action_z1.to(device)
            reward = reward.to(device)
            weighted_q, _ = ensemble_model(state_z0, state_z1, action_z0, action_z1)
            loss = loss_fn(weighted_q, reward)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * state_z0.size(0)
        train_loss /= len(train_loader.dataset)

        # Evaluate on validation set.
        ensemble_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for state_z0, state_z1, action_z0, action_z1, reward in val_loader:
                state_z0 = state_z0.to(device)
                state_z1 = state_z1.to(device)
                action_z0 = action_z0.to(device)
                action_z1 = action_z1.to(device)
                reward = reward.to(device)
                weighted_q, _ = ensemble_model(state_z0, state_z1, action_z0, action_z1)
                loss = loss_fn(weighted_q, reward)
                val_loss += loss.item() * state_z0.size(0)
        val_loss /= len(val_loader.dataset)

        tqdm.write(f"Logits Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_logits = ensemble_model.ensemble_logits.detach().clone()

    # Restore best ensemble_logits
    if best_logits is not None:
        ensemble_model.ensemble_logits.data.copy_(best_logits)
    return ensemble_model

def main():
    parser = argparse.ArgumentParser(description="Train QNet Ensemble and Fine-Tune Ensemble Logits")
    parser.add_argument('--ensemble-size', type=int, default=5,
                        help='Number of ensemble members to train')
    parser.add_argument('--base-model-name', type=str, default='cnn_qnet_member',
                        help='Base model name for individual ensemble members')
    parser.add_argument('--ensemble-model-name', type=str, default='ensemble_qnet',
                        help='Model name for saving the ensemble')
    parser.add_argument('--folder-name', type=str, default='ensemble',
                        help='Folder name where individual models are saved')
    parser.add_argument('--logits-epochs', type=int, default=20,
                        help='Number of epochs to train ensemble logits')
    args = parser.parse_args()
    
    ensemble_size = args.ensemble_size
    base_model_name = args.base_model_name
    ensemble_model_name = args.ensemble_model_name
    folder_name = args.folder_name

    ensemble_state_dicts = []

    # Train each ensemble member by calling QNetTrain.py
    for i in range(ensemble_size):
        member_model_name = f"{base_model_name}{i}"
        print(f"\nTraining ensemble member {i+1}/{ensemble_size} with model name '{member_model_name}'")
        result = subprocess.run([
            "python", "QNetTrain.py",
            "--model-name", member_model_name,
            "--folder-name", folder_name
        ])
        if result.returncode != 0:
            print(f"Training failed for ensemble member {i}. Exiting.")
            return
        
        # Locate the latest saved model file for the ensemble member.
        model_path = get_latest_model_path(folder_name, member_model_name)
        print(f"Loaded model for ensemble member {i} from: {model_path}")
        state_dict = torch.load(model_path)
        ensemble_state_dicts.append(state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble_model = QNetEnsemble(ensemble_size=ensemble_size, device=device).to(device)
    ensemble_model.load_state_dicts(ensemble_state_dicts)
    
    # Load training data for fine-tuning ensemble logits.
    print("\nLoading training data for ensemble logits fine-tuning...")
    data = load_data("small_zone_1", "training_data300")
    print(f"Loaded {len(data)} samples.")
    
    states_zone0 = torch.tensor(np.array([sample[0] for sample in data]), dtype=torch.float32).unsqueeze(1)
    states_zone1 = torch.tensor(np.array([sample[1] for sample in data]), dtype=torch.float32).unsqueeze(1)
    actions_zone0 = torch.tensor(np.array([sample[2] for sample in data]), dtype=torch.float32).unsqueeze(1)
    actions_zone1 = torch.tensor(np.array([sample[3] for sample in data]), dtype=torch.float32).unsqueeze(1)
    rewards = torch.tensor(np.array([sample[4] for sample in data]), dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(states_zone0, states_zone1, actions_zone0, actions_zone1, rewards)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Fine-tune only the ensemble logits.
    print("\nFine-tuning ensemble logits...")
    ensemble_model = train_ensemble_logits(ensemble_model, train_loader, val_loader, args.logits_epochs, device)
    
    # Save the final ensemble with trained logits.
    ensemble_save_prefix = os.path.join(MODEL_DIR, f"{ensemble_model_name}_ensemble_trained")
    ensemble_model.save_state_dicts(ensemble_save_prefix)
    print("\nEnsemble logits training complete. Ensemble models with trained logits saved.")

if __name__ == "__main__":
    main()
