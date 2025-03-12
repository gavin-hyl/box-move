"""
Benchmark script for comparing different policies on the BoxMoveEnvGym:
1. DQN Policy (trained model)
2. Random Policy (random valid actions)
3. Greedy Policy (select highest value box and place it in the minx, miny corner)
"""

import os
import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import environment and model components
from src.BoxMoveEnvGym import BoxMoveEnvGym
from src.QNet import CNNQNetwork
from src.Box import Box
from src.BoxAction import BoxAction
from src.Constants import ZONE0, ZONE1


def ensure_5d_tensor(tensor):
    """
    Ensure a tensor has 5 dimensions by adding missing dimensions.
    Expected: [batch, channel, depth, height, width]
    """
    if tensor.dim() < 4:
        tensor = tensor.unsqueeze(0)  # Add batch dimension
    if tensor.dim() < 5:
        tensor = tensor.unsqueeze(1)  # Add channel dimension
    return tensor


class RandomPolicy:
    """A policy that selects actions randomly from the valid actions."""
    
    def __init__(self, env):
        self.env = env
        self.name = "Random Policy"
    
    def select_action(self, obs):
        """
        Select a random valid action from the environment.
        
        Args:
            obs: The current observation (not used for random policy)
            
        Returns:
            action_idx: The index of the selected action
        """
        valid_actions = self.env.actions()
        
        if not valid_actions:
            # No valid actions, return a random action
            return random.randrange(self.env.action_space.n)
        
        # Select a random valid action
        random_action = random.choice(valid_actions)
        
        # Convert to action index
        action_idx = None
        for idx, act in self.env._action_map.items():
            if act == random_action:
                action_idx = idx
                break
        
        if action_idx is None:
            action_idx = random.randrange(self.env.action_space.n)
        
        return action_idx


class GreedyPolicy:
    """
    A greedy policy that selects the box with the highest value and
    places it in the minimum x, minimum y corner of zone1.
    """
    
    def __init__(self, env):
        self.env = env
        self.name = "Greedy Policy"
    
    def select_action(self, obs):
        """
        Select the action that moves the highest-value box to the min-corner.
        
        Args:
            obs: The current observation (not used for greedy policy)
            
        Returns:
            action_idx: The index of the selected action
        """
        valid_actions = self.env.actions()
        
        if not valid_actions:
            # No valid actions, return a random action
            return random.randrange(self.env.action_space.n)
        
        # Find boxes in zone0 that can be moved
        movable_boxes = []
        for i, box in enumerate(self.env.boxes):
            if box.zone == 0 and box.top_face() <= self.env.zone0_top:
                movable_boxes.append((i, box))
        
        if not movable_boxes:
            # No movable boxes, select random action
            random_action = random.choice(valid_actions)
            action_idx = None
            for idx, act in self.env._action_map.items():
                if act == random_action:
                    action_idx = idx
                    break
            
            if action_idx is None:
                action_idx = random.randrange(self.env.action_space.n)
            
            return action_idx
        
        # Find the box with the highest value
        highest_value_box = max(movable_boxes, key=lambda x: x[1].val)
        _, box = highest_value_box
        
        # Find the minimum x, minimum y corner in zone1 that's available
        min_x, min_y = float('inf'), float('inf')
        min_corner = None
        
        for pos in self.env.zone1_top:
            x, y, z = pos
            if x < min_x or (x == min_x and y < min_y):
                # Check if the box would fit here
                new_box = Box(np.array(pos), box.size, 1, 0)
                if (new_box.bottom_face() <= self.env.zone1_top and 
                    (np.array(pos) + box.size)[2] <= ZONE1[2]):
                    min_x, min_y = x, y
                    min_corner = pos
        
        if min_corner is None:
            # No suitable corner found, select random action
            random_action = random.choice(valid_actions)
            action_idx = None
            for idx, act in self.env._action_map.items():
                if act == random_action:
                    action_idx = idx
                    break
            
            if action_idx is None:
                action_idx = random.randrange(self.env.action_space.n)
            
            return action_idx
        
        # Create the action to move the box to the min corner
        greedy_action = BoxAction(tuple(box.pos), min_corner)
        
        # Convert to action index
        action_idx = None
        for idx, act in self.env._action_map.items():
            if act == greedy_action:
                action_idx = idx
                break
        
        if action_idx is None:
            # Fallback to random if the exact action isn't available
            random_action = random.choice(valid_actions)
            action_idx = None
            for idx, act in self.env._action_map.items():
                if act == random_action:
                    action_idx = idx
                    break
            
            if action_idx is None:
                action_idx = random.randrange(self.env.action_space.n)
        
        return action_idx


class DQNPolicy:
    """A policy that uses a trained DQN model to select actions."""
    
    def __init__(self, env, model_path, device="cpu"):
        self.env = env
        self.name = "DQN Policy"
        
        # Set device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Try to load the model - first try with CNNQNetwork, then with DuelingQNetwork
        try:
            # First attempt with CNNQNetwork
            from src.QNet import CNNQNetwork
            self.q_network = CNNQNetwork().to(self.device)
            
            # Load model
            print(f"Attempting to load model with CNNQNetwork architecture")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
            self.q_network.eval()
            print(f"Model loaded from {model_path} using CNNQNetwork architecture")
        except Exception as e1:
            print(f"Error loading with CNNQNetwork: {e1}")
            try:
                # Second attempt with DuelingQNetwork
                from src.QNet import DuelingQNetwork
                self.q_network = DuelingQNetwork().to(self.device)
                
                # Load model
                print(f"Attempting to load model with DuelingQNetwork architecture")
                checkpoint = torch.load(model_path, map_location=self.device)
                self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
                self.q_network.eval()
                print(f"Model loaded from {model_path} using DuelingQNetwork architecture")
            except Exception as e2:
                print(f"Error loading with DuelingQNetwork: {e2}")
                # Initialize with random weights if loading fails
                print("Using randomly initialized DQN model")
                from src.QNet import CNNQNetwork
                self.q_network = CNNQNetwork().to(self.device)
    
    def select_action(self, obs):
        """
        Select an action using the trained DQN model.
        
        Args:
            obs: Current observation from the environment
            
        Returns:
            action_idx: The index of the selected action
        """
        # Convert observation to tensor
        state_zone0 = torch.FloatTensor(obs["state_zone0"]).to(self.device)
        state_zone1 = torch.FloatTensor(obs["state_zone1"]).to(self.device)
        
        # Ensure 5D tensors
        state_zone0 = ensure_5d_tensor(state_zone0)
        state_zone1 = ensure_5d_tensor(state_zone1)
        
        # Get valid actions
        valid_actions = self.env.actions()
        
        if not valid_actions:
            # No valid actions, return a random action
            return random.randrange(self.env.action_space.n)
        
        # Calculate Q-values for all valid actions
        q_values = []
        action_indices = []
        
        for action in valid_actions:
            # Get action representation
            action_3d = self.env.action_3d(action)
            action_zone0 = torch.FloatTensor(action_3d[0]).to(self.device)
            action_zone1 = torch.FloatTensor(action_3d[1]).to(self.device)
            
            # Ensure 5D tensors
            action_zone0 = ensure_5d_tensor(action_zone0)
            action_zone1 = ensure_5d_tensor(action_zone1)
            
            # Get Q-value from network
            with torch.no_grad():
                q_value = self.q_network(state_zone0, state_zone1, action_zone0, action_zone1)
            
            # Find corresponding action index
            action_idx = None
            for idx, act in self.env._action_map.items():
                if act == action:
                    action_idx = idx
                    break
            
            if action_idx is not None:
                q_values.append(q_value.item())
                action_indices.append(action_idx)
        
        # Choose action with highest Q-value
        if q_values:
            best_idx = np.argmax(q_values)
            action_idx = action_indices[best_idx]
        else:
            # Fallback to random
            action_idx = random.randrange(self.env.action_space.n)
        
        return action_idx


def evaluate_policy(env, policy, num_episodes=100, seed=None, render=False):
    """
    Evaluate a policy on the environment over multiple episodes.
    
    Args:
        env: The environment to evaluate on
        policy: The policy to evaluate
        num_episodes: Number of episodes to run
        seed: Random seed for reproducibility
        render: Whether to render the environment during evaluation
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        env.reset(seed=seed)
    
    rewards = []
    episode_lengths = []
    occupancies = []
    execution_times = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False
        
        start_time = time.time()
        
        while not (terminated or truncated):
            # Select action using policy
            action_idx = policy.select_action(obs)
            
            # Take step in environment
            obs, reward, terminated, truncated, _ = env.step(action_idx)
            episode_reward += reward
            episode_length += 1
            
            # Render if requested
            if render and episode == 0:  # Only render first episode
                env.render()
                plt.pause(0.1)
        
        execution_time = time.time() - start_time
        
        # Record metrics
        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        occupancies.append(env.occupancy())
        execution_times.append(execution_time)
        
        if (episode + 1) % 10 == 0:
            print(f"Progress: {episode + 1}/{num_episodes} episodes")
    
    # Calculate metrics
    metrics = {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "mean_occupancy": np.mean(occupancies),
        "std_occupancy": np.std(occupancies),
        "mean_execution_time": np.mean(execution_times),
        "rewards": rewards,
        "episode_lengths": episode_lengths,
        "occupancies": occupancies
    }
    
    return metrics


def plot_comparison(metrics_dict, title="Policy Comparison", save_path=None):
    """
    Create comparison plots for different policies.
    
    Args:
        metrics_dict: Dictionary mapping policy names to their metrics
        title: Title for the plots
        save_path: Path to save the plots
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # Colors for different policies
    colors = ["blue", "green", "red", "purple", "orange"]
    
    # Plot rewards
    ax = axs[0]
    for i, (policy_name, metrics) in enumerate(metrics_dict.items()):
        mean_reward = metrics["mean_reward"]
        std_reward = metrics["std_reward"]
        color = colors[i % len(colors)]
        ax.bar(i, mean_reward, yerr=std_reward, color=color, alpha=0.7, label=policy_name)
    
    ax.set_xlabel("Policy")
    ax.set_ylabel("Average Reward")
    ax.set_title("Average Reward by Policy")
    ax.set_xticks(range(len(metrics_dict)))
    ax.set_xticklabels([])  # We'll use a shared legend instead
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Plot episode lengths
    ax = axs[1]
    for i, (policy_name, metrics) in enumerate(metrics_dict.items()):
        mean_length = metrics["mean_length"]
        std_length = metrics["std_length"]
        color = colors[i % len(colors)]
        ax.bar(i, mean_length, yerr=std_length, color=color, alpha=0.7, label=policy_name)
    
    ax.set_xlabel("Policy")
    ax.set_ylabel("Average Episode Length")
    ax.set_title("Average Episode Length by Policy")
    ax.set_xticks(range(len(metrics_dict)))
    ax.set_xticklabels([])  # We'll use a shared legend instead
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Plot occupancies
    ax = axs[2]
    for i, (policy_name, metrics) in enumerate(metrics_dict.items()):
        mean_occupancy = metrics["mean_occupancy"]
        std_occupancy = metrics["std_occupancy"]
        color = colors[i % len(colors)]
        ax.bar(i, mean_occupancy, yerr=std_occupancy, color=color, alpha=0.7, label=policy_name)
    
    ax.set_xlabel("Policy")
    ax.set_ylabel("Average Final Occupancy")
    ax.set_title("Average Final Occupancy by Policy")
    ax.set_xticks(range(len(metrics_dict)))
    ax.set_xticklabels([name for name in metrics_dict.keys()], rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add a shared legend at the bottom
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), ncol=len(metrics_dict))
    
    plt.suptitle(title, size=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved comparison plot to {save_path}")
    
    plt.show()


def main():
    # Create results directory
    os.makedirs("benchmark_results", exist_ok=True)
    
    # Environment parameters
    horizon = 50
    n_boxes = 5
    seed = 42
    
    # Evaluation parameters
    num_episodes = 50
    render = False  # Set to True to visualize the policies
    
    # Create environment
    env = BoxMoveEnvGym(horizon=horizon, n_boxes=n_boxes, seed=seed)
    
    # Create policies
    policies = {
        "Random": RandomPolicy(env),
        "Greedy": GreedyPolicy(env),
        "DQN": DQNPolicy(env, model_path="models/box_move_dqn.pt")
    }
    
    # Benchmark results
    results = {}
    
    # Evaluate each policy
    for name, policy in policies.items():
        print(f"\nEvaluating {name} policy...")
        metrics = evaluate_policy(
            env=env,
            policy=policy,
            num_episodes=num_episodes,
            seed=seed,
            render=render
        )
        
        results[policy.name] = metrics
        
        print(f"{name} Policy Results:")
        print(f"  Average Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Average Episode Length: {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")
        print(f"  Average Final Occupancy: {metrics['mean_occupancy']:.2f} ± {metrics['std_occupancy']:.2f}")
        print(f"  Average Execution Time: {metrics['mean_execution_time']:.4f} seconds")
    
    # Save detailed results
    np.save("benchmark_results/benchmark_data.npy", results)
    
    # Plot comparison
    plot_comparison(
        results,
        title=f"Policy Comparison (n_boxes={n_boxes}, horizon={horizon})",
        save_path="benchmark_results/policy_comparison.png"
    )
    
    # Generate detailed performance report
    with open("benchmark_results/benchmark_report.txt", "w") as f:
        f.write("Box Mover Policy Benchmark Report\n")
        f.write("================================\n\n")
        f.write(f"Environment Parameters:\n")
        f.write(f"- Horizon: {horizon}\n")
        f.write(f"- Number of Boxes: {n_boxes}\n")
        f.write(f"- Zone 0 Size: {ZONE0}\n")
        f.write(f"- Zone 1 Size: {ZONE1}\n")
        f.write(f"- Evaluation Episodes: {num_episodes}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write("------------------\n\n")
        
        for name, metrics in results.items():
            f.write(f"{name}:\n")
            f.write(f"  Reward:     {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}\n")
            f.write(f"  Length:     {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}\n")
            f.write(f"  Occupancy:  {metrics['mean_occupancy']:.2f} ± {metrics['std_occupancy']:.2f}\n")
            f.write(f"  Exec Time:  {metrics['mean_execution_time']:.4f} seconds\n\n")
        
        # Compute relative performance
        best_reward = max([m["mean_reward"] for m in results.values()])
        best_occupancy = max([m["mean_occupancy"] for m in results.values()])
        
        f.write("Relative Performance:\n")
        f.write("--------------------\n\n")
        
        for name, metrics in results.items():
            reward_ratio = metrics["mean_reward"] / best_reward if best_reward != 0 else 0
            occupancy_ratio = metrics["mean_occupancy"] / best_occupancy if best_occupancy != 0 else 0
            
            f.write(f"{name}:\n")
            f.write(f"  Reward:     {reward_ratio:.2%} of best\n")
            f.write(f"  Occupancy:  {occupancy_ratio:.2%} of best\n\n")
        
        f.write("\nDetailed Statistics:\n")
        f.write("-------------------\n\n")
        
        for name, metrics in results.items():
            f.write(f"{name}:\n")
            f.write(f"  Reward distribution:    min={min(metrics['rewards']):.2f}, max={max(metrics['rewards']):.2f}\n")
            f.write(f"  Length distribution:    min={min(metrics['episode_lengths'])}, max={max(metrics['episode_lengths'])}\n")
            f.write(f"  Occupancy distribution: min={min(metrics['occupancies']):.2f}, max={max(metrics['occupancies']):.2f}\n\n")
    
    print("\nBenchmark completed!")
    print("Results saved to benchmark_results/")


if __name__ == "__main__":
    main()