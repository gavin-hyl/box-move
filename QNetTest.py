import time
import numpy as np
import torch
import os
import argparse

from BoxMoveEnvGym import BoxMoveEnvGym
from QNet import CNNQNetwork
from QNetEnsemble import QNetEnsemble
from Constants import MODEL_DIR


def get_discrete_action_index(env, chosen_action):
    """
    Helper function to get the discrete action index for a given BoxAction.
    """
    for idx, act in env._action_map.items():
        if act == chosen_action:
            return idx
    return None


def run_episode_policy(env, policy, net=None, render=False):
    """
    Runs a single episode using a specified policy.

    Args:
        env (BoxMoveEnvGym): The environment instance.
        policy (str): Policy to use: "cnn", "random", or "greedy".
        net (CNNQNetwork): CNN Q-network (required if policy=="cnn").
        render (bool): If True, render the environment at each step.

    Returns:
        tuple: (total_reward, steps) obtained during the episode.
    """
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        valid_actions = env.env.actions()
        if not valid_actions:
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
                    q_value = net(state_zone0, state_zone1, action_zone0, action_zone1)
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
                candidate_actions = [
                    action for action in valid_actions 
                    if action.pos_from == tuple(best_box.pos)
                ]
                chosen_action = (
                    min(candidate_actions, key=lambda a: (a.pos_to[0], a.pos_to[1]))
                    if candidate_actions else np.random.choice(valid_actions)
                )
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


def benchmark_policy(policy, num_episodes=50, net=None):
    """
    Benchmarks a given policy over multiple episodes.

    Args:
        policy (str): Policy to use: "cnn", "random", or "greedy".
        num_episodes (int): Number of episodes to run.
        net (CNNQNetwork): CNN Q-network (required for "cnn" policy).
    """
    env = BoxMoveEnvGym(horizon=50, n_boxes=10)
    rewards = []
    for ep in range(num_episodes):
        total_reward, steps = run_episode_policy(env, policy, net=net)
        rewards.append(total_reward)
        print(f"[{policy.upper()}] Episode {ep+1}/{num_episodes}: Reward = {total_reward:.4f}, Steps = {steps}")
    avg_reward = np.mean(rewards)
    print(f"[{policy.upper()}] Mean Reward = {avg_reward:.4f}, Variance = {np.var(rewards):.4f}\n")


def load_ensemble_model(ensemble_model_prefix, ensemble_size, device):
    """
    Loads an ensemble model from files with naming pattern:
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
    Runs one episode using an ensemble Q-network.

    Args:
        env (BoxMoveEnvGym): The environment instance.
        ensemble_model (QNetEnsemble): Ensemble Q-network.
        render (bool): If True, render the environment.

    Returns:
        tuple: (total_reward, steps) from the episode.
    """
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        valid_actions = env.env.actions()
        if not valid_actions:
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
                q_val, _ = ensemble_model(state_zone0, state_zone1, action_zone0, action_zone1)
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
    Benchmarks the ensemble model over multiple episodes.

    Args:
        num_episodes (int): Number of episodes to run.
        ensemble_model (QNetEnsemble): The ensemble Q-network.
    """
    env = BoxMoveEnvGym(horizon=50, n_boxes=10)
    rewards = []
    for ep in range(num_episodes):
        total_reward, steps = run_episode_ensemble(env, ensemble_model)
        rewards.append(total_reward)
        print(f"[ENSEMBLE] Episode {ep+1}/{num_episodes}: Reward = {total_reward:.4f}, Steps = {steps}")
    avg_reward = np.mean(rewards)
    print(f"[ENSEMBLE] Mean Reward = {avg_reward:.4f}, Variance = {np.var(rewards):.4f}\n")


def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark all QNet policies in BoxMoveEnvGym: random, greedy, cnn, and ensemble."
    )
    parser.add_argument(
        "--num-episodes", type=int, default=20,
        help="Number of episodes to run for each benchmark."
    )
    parser.add_argument(
        "--model-path", type=str,
        default=os.path.join(MODEL_DIR, "vanilla", "cnn_qnet_epoch45.pth"),
        help="Path to the CNN Q-network model file (used for 'cnn' policy)."
    )
    parser.add_argument(
        "--ensemble-size", type=int, default=5,
        help="Number of ensemble members (used for 'ensemble' policy)."
    )
    parser.add_argument(
        "--ensemble-prefix", type=str,
        default=os.path.join(MODEL_DIR, "ensemble_qnet_ensemble_trained"),
        help="Prefix for ensemble model files (used for 'ensemble' policy)."
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render the environment during benchmarking."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("\n=== Benchmarking RANDOM Policy ===")
    benchmark_policy("random", num_episodes=args.num_episodes)

    print("\n=== Benchmarking GREEDY Policy ===")
    benchmark_policy("greedy", num_episodes=args.num_episodes)

    print("\n=== Benchmarking CNN Policy ===")
    net = CNNQNetwork()
    net.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    net.eval()
    benchmark_policy("cnn", num_episodes=args.num_episodes, net=net)

    print("\n=== Benchmarking ENSEMBLE Policy ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    ensemble_model = load_ensemble_model(args.ensemble_prefix, args.ensemble_size, device)
    print("Loaded ensemble model. Ensemble weights (softmax):",
          torch.softmax(ensemble_model.ensemble_logits, dim=0))
    benchmark_ensemble(num_episodes=args.num_episodes, ensemble_model=ensemble_model)


if __name__ == "__main__":
    main()
