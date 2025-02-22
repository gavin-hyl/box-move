import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from BoxMoveEnvGym import BoxMoveEnvGym
from QNet import CNNQNetwork
from Constants import MODEL_DIR, MODEL_NAME

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
        policy (str): "cnn" to use the CNN Q-network or "random" for random selection.
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
            # No valid actions remain.
            break

        if policy == "cnn":
            # Use the CNN to evaluate each valid action.
            state_3d = env.env.state_3d()  # Returns [zone0_dense, zone1_dense]
            state_zone0 = torch.tensor(state_3d[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            state_zone1 = torch.tensor(state_3d[1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            best_q = -float('inf')
            best_action = None
            
            for action in valid_actions:
                # Get the 3D representation of the action.
                action_3d = env.env.action_3d(action)
                action_zone0 = torch.tensor(action_3d[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                action_zone1 = torch.tensor(action_3d[1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    q_value = net.forward_separate(state_zone0, state_zone1, action_zone0, action_zone1)
                if q_value.item() > best_q:
                    best_q = q_value.item()
                    best_action = action

            chosen_action = best_action

        elif policy == "random":
            # Select a valid action at random.
            chosen_action = np.random.choice(valid_actions)
        else:
            raise ValueError("Unknown policy specified: choose 'cnn' or 'random'.")

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
        policy (str): "cnn" or "random".
        num_episodes (int): Number of episodes to run.
        net: CNN Q-network (required for "cnn" policy).
        
    Returns:
        avg_reward (float): Average total reward over episodes.
        avg_steps (float): Average number of steps per episode.
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
    return avg_reward, avg_steps

def main():
    num_episodes = 10

    # Benchmark random action selection policy.
    print("\nBenchmarking Random policy...")
    random_avg_reward, random_avg_steps = benchmark("random", num_episodes=num_episodes)
    print("\nRandom Policy:")
    print(f"  Average Reward: {random_avg_reward:.4f}")
    print(f"  Average Steps: {random_avg_steps:.2f}")

    # Benchmark CNN-based policy.
    print("Benchmarking CNN-based policy...")
    net = CNNQNetwork()
    model_path = f"{MODEL_DIR}/{MODEL_NAME}"
    net.load_state_dict(torch.load(model_path))
    net.eval()
    cnn_avg_reward, cnn_avg_steps = benchmark("cnn", num_episodes=num_episodes, net=net)
    print("\nCNN Policy:")
    print(f"  Average Reward: {cnn_avg_reward:.4f}")
    print(f"  Average Steps: {cnn_avg_steps:.2f}")

if __name__ == "__main__":
    main()
