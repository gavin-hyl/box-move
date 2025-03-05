import time
import numpy as np
import torch

from BoxMoveEnvGym import BoxMoveEnvGym
from QNet import CNNQNetwork
from Constants import MODEL_DIR

import os
from QNetEnsemble import QNetEnsemble

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
    print(f"Mean = {avg_reward:.4f}, Var = {np.var(rewards):.4f}")

def main():
    num_episodes = 20

    print("\nBenchmarking Random policy...")
    benchmark("random", num_episodes=num_episodes)

    print("\nBenchmarking CNN-based policy...")
    net = CNNQNetwork()
    model_path = f"{MODEL_DIR}/vanilla/cnn_qnet_epoch45.pth"
    net.load_state_dict(torch.load(model_path))
    net.eval()
    benchmark("cnn", num_episodes=num_episodes, net=net)

    print("\nBenchmarking Greedy policy...")
    benchmark("greedy", num_episodes=num_episodes)


def load_ensemble_model(ensemble_model_prefix, ensemble_size, device):
    """
    Loads the ensemble model from files with naming pattern:
    {ensemble_model_prefix}_{i}.pth for i in range(ensemble_size).
    """
    state_dicts = []
    for i in range(ensemble_size):
        file_path = f"{ensemble_model_prefix}_{i}.pth"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Ensemble model file not found: {file_path}")
        state_dict = torch.load(file_path, map_location=device)
        state_dicts.append(state_dict)
    ensemble_model = QNetEnsemble(ensemble_size=ensemble_size, device=device).to(device)
    ensemble_model.load_state_dicts(state_dicts)
    ensemble_model.eval()
    return ensemble_model

def run_episode_ensemble(env, ensemble_model, render=False):
    """
    Runs one episode in the environment using the ensemble model.
    Uses the same logic as the CNN-based policy, but extracts the weighted Q-value
    from the ensemble's forward pass (which returns (weighted_q, variance)).
    """
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        valid_actions = env.env.actions()
        if len(valid_actions) == 0:
            break
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
                q_val, _ = ensemble_model.forward(state_zone0, state_zone1, action_zone0, action_zone1)
            if q_val.item() > best_q:
                best_q = q_val.item()
                best_action = action
        if best_action is None:
            break
        discrete_action = get_discrete_action_index(env, best_action)
        if discrete_action is None:
            print("Warning: Discrete action index not found. Skipping step.")
            break
        obs, reward, done, truncated, info = env.step(discrete_action)
        total_reward += reward
        steps += 1
        if render:
            env.render()
    return total_reward, steps

def benchmark_ensemble(num_episodes=20, ensemble_model=None):
    """
    Benchmarks the ensemble model over a number of episodes using a CNN-style policy.
    """
    env = BoxMoveEnvGym(horizon=50, n_boxes=10)
    rewards = []
    for ep in range(num_episodes):
        total_reward, steps = run_episode_ensemble(env, ensemble_model)
        rewards.append(total_reward)
        print(f"Ensemble Episode {ep+1}/{num_episodes}: Reward = {total_reward:.4f}, Steps = {steps}")
    avg_reward = np.mean(rewards)
    print(f"Ensemble Mean Reward = {avg_reward:.4f}, Variance = {np.var(rewards):.4f}")

def test_ensemble():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble_size = 5   # FIXME
    # Set the prefix used when saving the ensemble.
    ensemble_model_prefix = os.path.join(MODEL_DIR, "ensemble")
    ensemble_model = load_ensemble_model(ensemble_model_prefix, ensemble_size, device)
    print("Loaded ensemble model. Ensemble weights (softmax):", torch.softmax(ensemble_model.ensemble_logits, dim=0))
    benchmark_ensemble(num_episodes=20, ensemble_model=ensemble_model)

if __name__ == "__main__":
    # Run the original tests.
    main()
    print("\n\n=== Testing Ensemble Model ===\n")
    test_ensemble()
