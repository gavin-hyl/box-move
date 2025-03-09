import argparse
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import sys

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.BoxMoveEnvGym import BoxMoveEnvGym
from src.OptimizedDQN import OptimizedDQN

def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN agent on the BoxMoveEnv")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total timesteps to train")
    parser.add_argument("--horizon", type=int, default=30, help="Episode horizon")
    parser.add_argument("--n_boxes", type=int, default=4, help="Number of boxes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=50000, help="Replay buffer size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--train_freq", type=int, default=4, help="Train frequency")
    parser.add_argument("--target_update", type=int, default=1000, help="Target network update interval")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient")
    parser.add_argument("--exploration_fraction", type=float, default=0.2, help="Fraction of total timesteps for epsilon decay")
    parser.add_argument("--exploration_initial_eps", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--exploration_final_eps", type=float, default=0.05, help="Final exploration rate")
    parser.add_argument("--env_pool_size", type=int, default=4, help="Environment pool size")
    parser.add_argument("--eval_freq", type=int, default=2500, help="Evaluation frequency")
    parser.add_argument("--log_freq", type=int, default=500, help="Logging frequency")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--train", action="store_true", help="Train the DQN model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the trained model")
    parser.add_argument("--compare", action="store_true", help="Compare DQN to random policy")
    parser.add_argument("--quick-test", action="store_true", help="Quick test with simplified settings")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize environment
    env = BoxMoveEnvGym(
        horizon=args.horizon,
        n_boxes=args.n_boxes,
        seed=args.seed
    )
    
    # Create directories for results
    os.makedirs("results", exist_ok=True)
    
    # Initialize DQN agent
    dqn = OptimizedDQN(
        env=env,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        target_update_interval=args.target_update,
        tau=args.tau,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        device=args.device,
        seed=args.seed,
        env_pool_size=args.env_pool_size,
        double_q=False
    )
    
    if args.train:
        print(f"Training DQN for {args.timesteps} timesteps...")
        dqn.learn(
            total_timesteps=args.timesteps,
            eval_freq=args.eval_freq,
            log_freq=args.log_freq
        )
        # Save model and learning curve
        dqn.save("results/dqn_model.pt")
        dqn.plot_learning_curve(save_path="results/learning_curve.png")
        print("Training complete! Model saved to results/dqn_model.pt")
    
    if args.evaluate:
        # Try to load a pre-trained model if not trained in this session
        if not args.train:
            try:
                model_path = "results/dqn_model.pt"
                dqn.load(model_path)
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Could not load model: {e}")
                print("Training a new model for evaluation...")
                dqn.learn(total_timesteps=5000, log_freq=500)
        
        # Evaluate the agent
        print("\nEvaluating DQN performance...")
        mean_reward, mean_length = dqn.evaluate(num_episodes=10)
        print(f"Mean reward: {mean_reward:.2f}, Mean episode length: {mean_length:.2f}")
    
    if args.compare:
        compare_dqn_with_random(env, dqn)
    
    if args.quick_test:
        quick_test()

def quick_test():
    """Run a quick test with simplified settings"""
    print("\nRunning quick test with simplified settings...")
    
    # Initialize environment with simple settings
    env = BoxMoveEnvGym(
        horizon=20,
        n_boxes=8,
        seed=42
    )
    
    # Initialize DQN with simplified settings
    dqn = OptimizedDQN(
        env=env,
        learning_rate=3e-4,
        buffer_size=10000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        device="auto",
        seed=42,
        env_pool_size=4,
        double_q=False
    )
    
    # Train for a short period
    print("Training DQN for 5000 timesteps...")
    dqn.learn(total_timesteps=5000, eval_freq=2500, log_freq=500)
    
    # Save model and learning curve
    dqn.save("quick_test_model.pt")
    dqn.plot_learning_curve(save_path="quick_test_learning_curve.png")
    
    # Evaluate
    print("\nEvaluating model performance...")
    mean_reward, mean_length = dqn.evaluate(num_episodes=5)
    print(f"Evaluation results: Mean reward: {mean_reward:.2f}, Mean episode length: {mean_length:.2f}")
    print("Quick test complete! Model saved to quick_test_model.pt")

def compare_dqn_with_random(env=None, dqn=None, n_episodes=50, horizon=25, n_boxes=6, seed=42):
    """
    Compare DQN policy against a random policy
    """
    print("Comparing DQN policy against random policy (sampling from valid actions)...")
    
    # Initialize environment if not provided
    if env is None:
        env = BoxMoveEnvGym(horizon=horizon, n_boxes=n_boxes, seed=seed)
        print(f"Created environment with {n_boxes} boxes and horizon {horizon}")
    
    # Use provided DQN or create a new one
    if dqn is None:
        dqn = OptimizedDQN(
            env,
            learning_rate=3e-4,
            buffer_size=10000,
            batch_size=32,
            gamma=0.99,
            train_freq=4,
            target_update_interval=500,
            exploration_fraction=0.2,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            device="auto",
            seed=seed,
            env_pool_size=4,
            double_q=False
        )
        
        # Try to load a pre-trained model
        try:
            model_paths = ["results/dqn_model.pt", "quick_test_model.pt", "standard_dqn.pt"]
            model_loaded = False
            
            for path in model_paths:
                if os.path.exists(path):
                    print(f"Loading DQN from {path}")
                    dqn.load(path)
                    model_loaded = True
                    break
            
            if not model_loaded:
                print("No pre-trained model found. Training a new model...")
                dqn.learn(total_timesteps=5000, log_freq=500)
                dqn.save("results/dqn_model.pt")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training a new model...")
            dqn.learn(total_timesteps=5000, log_freq=500)
            dqn.save("results/dqn_model.pt")
    
    # Evaluate DQN policy
    print("\nEvaluating DQN policy...")
    dqn_rewards = []
    dqn_lengths = []
    
    # Add progress bar for DQN evaluation
    dqn_eval_progress = tqdm(range(n_episodes), desc="DQN Evaluation")
    
    for i in dqn_eval_progress:
        obs = env.reset(seed=seed+i)  # Different seed for each episode
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Get DQN action
            action, _ = dqn._select_action(obs, deterministic=True)
            
            # Take action in environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            
            if episode_length >= horizon:
                break
        
        dqn_rewards.append(episode_reward)
        dqn_lengths.append(episode_length)
        
        # Update progress bar description
        dqn_eval_progress.set_description(f"DQN Eval: Episode {i+1}/{n_episodes}, Reward: {episode_reward:.2f}")
    
    # Evaluate random policy
    print("\nEvaluating random policy (sampling from valid actions)...")
    random_rewards = []
    random_lengths = []
    
    # Add progress bar for random policy evaluation
    random_eval_progress = tqdm(range(n_episodes), desc="Random Policy Evaluation")
    
    for i in random_eval_progress:
        env.reset(seed=seed+i)  # Same seeds as DQN evaluation
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Random selection from valid actions
            valid_actions = env.actions()
            if not valid_actions:
                break
                
            # Select random action from valid actions
            action = random.choice(valid_actions)
            
            # Convert to action index if needed
            if hasattr(env, '_action_map'):
                for idx, act in env._action_map.items():
                    if act == action:
                        action_idx = idx
                        break
            else:
                action_idx = env.action_space.sample()
            
            # Take action in environment
            _, reward, terminated, truncated, _ = env.step(action_idx)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            
            if episode_length >= horizon:
                break
        
        random_rewards.append(episode_reward)
        random_lengths.append(episode_length)
        
        # Update progress bar description
        random_eval_progress.set_description(f"Random Eval: Episode {i+1}/{n_episodes}, Reward: {episode_reward:.2f}")
    
    # Calculate statistics
    dqn_mean_reward = np.mean(dqn_rewards)
    dqn_std_reward = np.std(dqn_rewards)
    dqn_mean_length = np.mean(dqn_lengths)
    dqn_success_rate = sum(1 for r in dqn_rewards if r > 0) / len(dqn_rewards) * 100
    
    random_mean_reward = np.mean(random_rewards)
    random_std_reward = np.std(random_rewards)
    random_mean_length = np.mean(random_lengths)
    random_success_rate = sum(1 for r in random_rewards if r > 0) / len(random_rewards) * 100
    
    # Print results
    print("\n--- Comparison Results ---")
    print("\nDQN Policy:")
    print(f"  Mean Reward: {dqn_mean_reward:.2f} ± {dqn_std_reward:.2f}")
    print(f"  Mean Episode Length: {dqn_mean_length:.2f}")
    print(f"  Success Rate: {dqn_success_rate:.2f}%")
    
    print("\nRandom Policy (Valid Actions):")
    print(f"  Mean Reward: {random_mean_reward:.2f} ± {random_std_reward:.2f}")
    print(f"  Mean Episode Length: {random_mean_length:.2f}")
    print(f"  Success Rate: {random_success_rate:.2f}%")
    
    # Calculate improvement percentage
    if random_mean_reward != 0:
        improvement = (dqn_mean_reward - random_mean_reward) / abs(random_mean_reward) * 100
        print(f"\nDQN vs Random: {improvement:.2f}% improvement in reward")
    else:
        print("\nCannot calculate percentage improvement (random reward is 0)")
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Plot episode rewards
    plt.subplot(2, 2, 1)
    plt.plot(dqn_rewards, label='DQN', alpha=0.7)
    plt.plot(random_rewards, label='Random', alpha=0.7)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot reward distributions
    plt.subplot(2, 2, 2)
    plt.hist(dqn_rewards, alpha=0.5, label='DQN', bins=10)
    plt.hist(random_rewards, alpha=0.5, label='Random', bins=10)
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot episode lengths
    plt.subplot(2, 2, 3)
    plt.plot(dqn_lengths, label='DQN', alpha=0.7)
    plt.plot(random_lengths, label='Random', alpha=0.7)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot success rate comparison
    plt.subplot(2, 2, 4)
    plt.bar(['DQN Policy', 'Random Policy'], [dqn_success_rate, random_success_rate], color=['blue', 'red'], alpha=0.7)
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('comparison_results.png')
    plt.close()
    
    # Save detailed results to file
    with open("comparison_results.txt", "w") as f:
        f.write("=== Box Move Environment Policy Comparison ===\n\n")
        f.write(f"Configuration: {n_episodes} episodes, {n_boxes} boxes, horizon {horizon}\n\n")
        
        f.write("--- DQN Policy ---\n")
        f.write(f"Mean Reward: {dqn_mean_reward:.2f} ± {dqn_std_reward:.2f}\n")
        f.write(f"Mean Episode Length: {dqn_mean_length:.2f}\n")
        f.write(f"Success Rate: {dqn_success_rate:.2f}%\n\n")
        
        f.write("--- Random Policy (Valid Actions) ---\n")
        f.write(f"Mean Reward: {random_mean_reward:.2f} ± {random_std_reward:.2f}\n")
        f.write(f"Mean Episode Length: {random_mean_length:.2f}\n")
        f.write(f"Success Rate: {random_success_rate:.2f}%\n\n")
        
        if random_mean_reward != 0:
            f.write(f"DQN vs Random: {improvement:.2f}% improvement in reward\n")
    
    print("Detailed results saved to comparison_results.txt")
    print("Visualization saved to comparison_results.png")

if __name__ == "__main__":
    args = parse_args()
    
    if args.quick_test:
        quick_test()
    elif args.compare:
        compare_dqn_with_random()
    else:
        main()
