import subprocess
import os
import glob
import torch
import argparse

from QNetEnsemble import QNetEnsemble
from Constants import MODEL_DIR

def get_latest_model_path(folder_name, model_name):
    """
    Searches MODEL_DIR for the latest model file matching the pattern
    {model_name}_epoch*.pth and returns its path.
    """
    pattern = os.path.join(MODEL_DIR, folder_name, f"{model_name}_epoch*.pth")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No model files found for {model_name} in {MODEL_DIR}/{folder_name}")
    
    def get_epoch(file_path):
        basename = os.path.basename(file_path)
        # Expected format: {model_name}_epoch{epoch}.pth
        try:
            epoch_str = basename.split("_epoch")[1].split(".pth")[0]
            return int(epoch_str)
        except (IndexError, ValueError):
            return -1

    # Pick the file with the highest epoch number
    latest_file = max(files, key=get_epoch)
    return latest_file

def main():
    parser = argparse.ArgumentParser(description="Train QNet Ensemble by calling QNetTrain for each member.")
    parser.add_argument('--ensemble-size', type=int, default=5,
                        help='Number of ensemble members to train')
    # Optionally pass num_epochs if QNetTrain is updated to accept it.
    parser.add_argument('--base-model-name', type=str, default='cnn_qnet_member',
                        help='Base model name for individual ensemble members')
    parser.add_argument('--ensemble-model-name', type=str, default='ensemble_qnet',
                        help='Model name for saving the ensemble')
    args = parser.parse_args()
    
    ensemble_size = args.ensemble_size
    base_model_name = args.base_model_name
    ensemble_model_name = args.ensemble_model_name

    ensemble_state_dicts = []

    # For each ensemble member, call QNetTrain.py via subprocess
    for i in range(ensemble_size):
        member_model_name = f"{base_model_name}{i}"
        print(f"\nTraining ensemble member {i+1} of {ensemble_size} with model name '{member_model_name}'")
        # Call QNetTrain.py with the desired arguments.
        result = subprocess.run([
            "python", "QNetTrain.py",
            "--model-name", member_model_name,
            "--folder-name", "ensemble",
        ])
        if result.returncode != 0:
            print(f"Training failed for ensemble member {i}. Exiting.")
            return
        
        # After training, locate the saved model file.
        model_path = get_latest_model_path(member_model_name)
        print(f"Loaded model for ensemble member {i} from: {model_path}")
        state_dict = torch.load(model_path)
        ensemble_state_dicts.append(state_dict)
    
    # Create the ensemble model and load the individual networks’ state dictionaries.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble_model = QNetEnsemble(ensemble_size=ensemble_size, device=device).to(device)
    ensemble_model.load_state_dicts(ensemble_state_dicts)
    
    # Save each network’s state in the ensemble.
    ensemble_save_prefix = os.path.join(MODEL_DIR, f"{ensemble_model_name}_ensemble")
    ensemble_model.save_state_dicts(ensemble_save_prefix)
    print("\nEnsemble training complete. Ensemble models saved.")

if __name__ == "__main__":
    main()
