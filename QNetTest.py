import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from BoxMoveEnvGym import BoxMoveEnvGym
from QNet import CNNQNetwork
from Constants import MODEL_DIR

def get_discrete_action_index(env, chosen_action):
    """Helper function to get the discrete action index for a given BoxAction."""
    for idx, act in env._action_map.items():
        if act == chosen_action:
            return idx
    return None

def run_episode(env, policy, net=None, render=False):
    """
    Runs a single episode in the environment using the specified policy.
    
    Args:
        env: Instance of BoxMoveEnvGym.
        policy (str): "cnn", "random", or "greedy".
        net: The CNN Q-network (required if policy=="cnn").
        render (bool): Whether to render the environment at each step.
        
    Returns:
        total_reward (float): Sum of rewards obtained in the episode.
        steps (int): Number of steps taken.
    """
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        valid_actions = env.env.actions()
        if len(valid_actions) == 0:
            break

        if policy == "cnn":
            state_3d = env.env.state_3d()  # [zone0_dense, zone1_dense]
            state_zone0 = torch.tensor(state_3d[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            state_zone1 = torch.tensor(state_3d[1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            best_q = -float('inf')
            best_action = None
            
            for action in valid_actions:
                action_3d = env.env.action_3d(action)
                action_zone0 = torch.tensor(action_3d[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                action_zone1 = torch.tensor(action_3d[1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    q_value = net.forward(state_zone0, state_zone1, action_zone0, action_zone1)
                if q_value.item() > best_q:
                    best_q = q_value.item()
                    best_action = action
            chosen_action = best_action

        elif policy == "random":
            chosen_action = np.random.choice(valid_actions)
        elif policy == "greedy":
            best_density = -float('inf')
            best_box = None
            for box in env.env.boxes:
                if box.zone == 0 and box.val_density() > best_density:
                    best_density = box.val_density()
                    best_box = box
            if best_box is None:
                chosen_action = np.random.choice(valid_actions)
            else:
                candidate_actions = [action for action in valid_actions if action.pos_from == tuple(best_box.pos)]
                if candidate_actions:
                    chosen_action = min(candidate_actions, key=lambda a: (a.pos_to[0], a.pos_to[1]))
                else:
                    chosen_action = np.random.choice(valid_actions)
        else:
            raise ValueError("Unknown policy specified: choose 'cnn', 'random', or 'greedy'.")

        discrete_action = get_discrete_action_index(env, chosen_action)
        if discrete_action is None:
            print("Warning: Discrete action index not found. Skipping step.")
            break

        obs, reward, done, truncated, info = env.step(discrete_action)
        total_reward += reward
        steps += 1

        if render:
            env.render()
            time.sleep(0.5)

    return total_reward, steps

def benchmark(policy, num_episodes=50, net=None):
    """
    Benchmarks a given policy over a number of episodes.
    
    Args:
        policy (str): "cnn", "random", or "greedy".
        num_episodes (int): Number of episodes to run.
        net: CNN Q-network (required for "cnn" policy).
        
    Returns:
        avg_reward (float): Average total reward over episodes.
        avg_steps (float): Average number of steps per episode.
        rewards (list): List of total rewards from each episode.
    """
    env = BoxMoveEnvGym(horizon=50, n_boxes=10)
    rewards = []
    steps_list = []
    for ep in range(num_episodes):
        total_reward, steps = run_episode(env, policy, net=net)
        rewards.append(total_reward)
        steps_list.append(steps)
        print(f"Episode {ep+1}/{num_episodes}: Reward = {total_reward:.4f}, Steps = {steps}")
    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps_list)
    return avg_reward, avg_steps, rewards

def draw_combined_histogram(rewards_dict, title="Reward Histogram", bins=10):
    """
    Draws a combined histogram for rewards from multiple policies.
    
    Args:
        rewards_dict (dict): Dictionary mapping policy names to lists of rewards.
        title (str): Title of the histogram.
        bins (int): Number of bins in the histogram.
    """
    plt.figure(figsize=(10, 7))
    for policy_name, rewards in rewards_dict.items():
        plt.hist(rewards, bins=bins, alpha=0.5, label=policy_name, edgecolor='black')
    plt.title(title)
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def main():
    num_episodes = 20
    rewards_dict = {}

    # Benchmark Random policy.
    print("\nBenchmarking Random policy...")
    random_avg_reward, random_avg_steps, random_rewards = benchmark("random", num_episodes=num_episodes)
    print(f"Random Policy: Avg Reward = {random_avg_reward:.4f}, Avg Steps = {random_avg_steps:.2f}")
    rewards_dict["Random"] = random_rewards

    # Benchmark CNN-based policy.
    print("\nBenchmarking CNN-based policy...")
    net = CNNQNetwork()
    model_path = f"{MODEL_DIR}/cnn_qnet_epoch45.pth"
    net.load_state_dict(torch.load(model_path))
    net.eval()
    cnn_avg_reward, cnn_avg_steps, cnn_rewards = benchmark("cnn", num_episodes=num_episodes, net=net)
    print(f"CNN Policy: Avg Reward = {cnn_avg_reward:.4f}, Avg Steps = {cnn_avg_steps:.2f}")
    rewards_dict["CNN"] = cnn_rewards

    # Benchmark Greedy policy.
    print("\nBenchmarking Greedy policy...")
    greedy_avg_reward, greedy_avg_steps, greedy_rewards = benchmark("greedy", num_episodes=num_episodes)
    print(f"Greedy Policy: Avg Reward = {greedy_avg_reward:.4f}, Avg Steps = {greedy_avg_steps:.2f}")
    rewards_dict["Greedy"] = greedy_rewards

    # Draw a combined histogram of rewards.
    draw_combined_histogram(rewards_dict, title="Combined Policy Reward Histogram", bins=10)
    
if __name__ == "__main__":
    main()
