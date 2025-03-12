import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.BoxMoveEnvGym import BoxMoveEnvGym
from src.QNet import CNNQNetwork
from src.Benchmark import RandomPolicy, GreedyPolicy

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a basic CNN for the BoxMove environment")
    parser.add_argument("--n_boxes", type=int, default=5, 
                        help="Number of boxes in the environment (default: 5)")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size for training (default: 64)")
    parser.add_argument("--lr", type=float, default=0.001, 
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--data_path", type=str, default="data/combined_tree_search_data.npz", 
                        help="Path to load the dataset (default: data/combined_tree_search_data.npz)")
    parser.add_argument("--model_path", type=str, default="models/simple_cnn_model.pt", 
                        help="Path to save the trained model (default: models/simple_cnn_model.pt)")
    parser.add_argument("--eval_episodes", type=int, default=30, 
                        help="Number of episodes for evaluation (default: 30)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed (default: 42)")
    return parser.parse_args()

def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_dataset(data_path):
    """
    Load the training dataset.
    
    Args:
        data_path: Path to the dataset file
        
    Returns:
        dataset: List of (state_zone0, state_zone1, action_zone0, action_zone1, value) tuples
    """
    print(f"Loading dataset from {data_path}...")
    
    # Load the arrays
    data = np.load(data_path)
    
    # Convert to pytorch tensors
    states_zone0 = torch.FloatTensor(data['states_zone0'])
    states_zone1 = torch.FloatTensor(data['states_zone1'])
    actions_zone0 = torch.FloatTensor(data['actions_zone0'])
    actions_zone1 = torch.FloatTensor(data['actions_zone1'])
    values = data['values']
    
    # Create the dataset
    dataset = []
    for i in range(len(values)):
        dataset.append((
            states_zone0[i],
            states_zone1[i],
            actions_zone0[i],
            actions_zone1[i],
            values[i]
        ))
    
    print(f"Loaded {len(dataset)} state-action-value pairs")
    return dataset

def train_cnn(dataset, args):
    """
    Train a CNN model on the dataset.
    
    Args:
        dataset: Training dataset
        args: Command line arguments
        
    Returns:
        cnn_model: Trained CNN model
    """
    print("Training CNN model...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    cnn_model = CNNQNetwork().to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=args.lr)
    
    # Ensure dataset is not empty
    if len(dataset) == 0:
        print("Error: Dataset is empty!")
        return None
    
    # Adjust batch size if necessary
    batch_size = min(args.batch_size, len(dataset))
    if batch_size < args.batch_size:
        print(f"Warning: Dataset size ({len(dataset)}) is smaller than batch size ({args.batch_size})")
        print(f"Adjusted batch size to {batch_size}")
    
    # Training loop
    batch_indices = list(range(len(dataset)))
    num_batches = max(1, len(dataset) // batch_size)
    
    # Lists to track metrics
    losses = []
    
    for epoch in range(args.epochs):
        random.shuffle(batch_indices)
        epoch_loss = 0.0
        
        for batch_idx in range(num_batches):
            optimizer.zero_grad()
            
            # Get batch
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(dataset))
            indices = batch_indices[batch_start:batch_end]
            
            # Prepare data
            states_zone0 = []
            states_zone1 = []
            actions_zone0 = []
            actions_zone1 = []
            values = []
            
            for i in indices:
                state_zone0, state_zone1, action_zone0, action_zone1, value = dataset[i]
                states_zone0.append(state_zone0)
                states_zone1.append(state_zone1)
                actions_zone0.append(action_zone0)
                actions_zone1.append(action_zone1)
                values.append(value)
            
            # Prepare batch tensors
            states_zone0 = torch.stack(states_zone0).to(device)
            states_zone1 = torch.stack(states_zone1).to(device)
            actions_zone0 = torch.cat(actions_zone0).to(device)
            actions_zone1 = torch.cat(actions_zone1).to(device)
            target_values = torch.FloatTensor(values).unsqueeze(1).to(device)
            
            # Forward pass
            q_values = cnn_model(states_zone0, states_zone1, actions_zone0, actions_zone1)
            
            # Compute loss
            loss = criterion(q_values, target_values)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Track average loss for this epoch
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('results/simple_training_loss.png')
    
    # Save model
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    torch.save(cnn_model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")
    
    return cnn_model

class CNNPolicy:
    """Policy that uses the trained CNN model."""
    
    def __init__(self, env, cnn_model, device):
        self.env = env
        self.cnn_model = cnn_model
        self.device = device
        self.name = "CNN Policy"
    
    def select_action(self, obs):
        # Get valid actions
        valid_actions = self.env.actions()
        
        if not valid_actions:
            # No valid actions, return a random action
            return random.randrange(self.env.action_space.n)
        
        # Get state tensors
        state_zone0 = torch.FloatTensor(obs["state_zone0"]).to(self.device)
        state_zone1 = torch.FloatTensor(obs["state_zone1"]).to(self.device)
        
        # Evaluate each valid action
        best_action = None
        best_value = float('-inf')
        
        for action in valid_actions:
            # Get action tensors
            action_3d = self.env.action_3d(action)
            action_zone0 = torch.FloatTensor(action_3d[0]).unsqueeze(0).unsqueeze(0).to(self.device)
            action_zone1 = torch.FloatTensor(action_3d[1]).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Evaluate action
            with torch.no_grad():
                q_value = self.cnn_model(state_zone0, state_zone1, action_zone0, action_zone1).item()
            
            # Update best action
            if q_value > best_value:
                best_value = q_value
                best_action = action
        
        # Convert to action index
        action_idx = None
        for idx, act in self.env._action_map.items():
            if act == best_action:
                action_idx = idx
                break
        
        return action_idx

def evaluate_policy(env, policy, num_episodes=30):
    """
    Evaluate a policy on the environment.
    
    Args:
        env: The BoxMoveEnvGym environment
        policy: The policy to evaluate
        num_episodes: Number of episodes to evaluate
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    rewards = []
    episode_lengths = []
    
    for _ in tqdm(range(num_episodes), desc=f"Evaluating {policy.name}"):
        total_reward = 0
        episode_length = 0
        obs = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            action = policy.select_action(obs)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            episode_length += 1
        
        rewards.append(total_reward)
        episode_lengths.append(episode_length)
    
    # Calculate metrics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_length = np.mean(episode_lengths)
    
    metrics = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "rewards": rewards,
        "lengths": episode_lengths
    }
    
    return metrics

def benchmark_policies(cnn_model, args):
    """
    Benchmark different policies.
    
    Args:
        cnn_model: Trained CNN model
        args: Command line arguments
    """
    print("Benchmarking policies...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize environment
    env = BoxMoveEnvGym(horizon=args.n_boxes, n_boxes=args.n_boxes)
    
    # Initialize policies
    policies = [
        RandomPolicy(env),
        GreedyPolicy(env),
        CNNPolicy(env, cnn_model, device)
    ]
    
    # Evaluate policies
    metrics = {}
    
    for policy in policies:
        policy_metrics = evaluate_policy(env, policy, num_episodes=args.eval_episodes)
        metrics[policy.name] = policy_metrics
        
        print(f"\n{policy.name} metrics:")
        print(f"  Mean reward: {policy_metrics['mean_reward']:.2f} ± {policy_metrics['std_reward']:.2f}")
        print(f"  Mean episode length: {policy_metrics['mean_length']:.2f}")
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Box plot of rewards
    plt.subplot(1, 2, 1)
    box_data = [metrics[policy.name]["rewards"] for policy in policies]
    plt.boxplot(box_data, labels=[policy.name for policy in policies])
    plt.title("Reward Distribution")
    plt.ylabel("Reward")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Bar plot of mean rewards
    plt.subplot(1, 2, 2)
    means = [metrics[policy.name]["mean_reward"] for policy in policies]
    stds = [metrics[policy.name]["std_reward"] for policy in policies]
    bars = plt.bar(range(len(policies)), means, yerr=stds, capsize=10)
    plt.xticks(range(len(policies)), [policy.name for policy in policies])
    plt.title("Mean Rewards")
    plt.ylabel("Mean Reward")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/simple_benchmark_comparison.png")
    
    # Save metrics to file
    with open("results/simple_benchmark_metrics.txt", "w") as f:
        for policy_name, policy_metrics in metrics.items():
            f.write(f"{policy_name}:\n")
            f.write(f"  Mean reward: {policy_metrics['mean_reward']:.2f} ± {policy_metrics['std_reward']:.2f}\n")
            f.write(f"  Mean episode length: {policy_metrics['mean_length']:.2f}\n\n")
    
    return metrics

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Load dataset
    dataset = load_dataset(args.data_path)
    
    # Train CNN model
    cnn_model = train_cnn(dataset, args)
    
    # Benchmark policies
    metrics = benchmark_policies(cnn_model, args)
    
    print("\nTraining and evaluation complete!")
    print("Results saved to results/simple_benchmark_comparison.png")
    print("Metrics saved to results/simple_benchmark_metrics.txt")
    print(f"Model saved to {args.model_path}")

if __name__ == "__main__":
    main() 