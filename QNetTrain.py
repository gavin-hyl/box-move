import os
import time
import argparse
import torch
import psutil
import cProfile
import pstats
import random
import matplotlib.pyplot as plt
import numpy as np

from BoxMoveEnvGym import BoxMoveEnvGym
from optimized_dqn import OptimizedDQN  # Ensure this file is in the same directory

def parse_args():
    parser = argparse.ArgumentParser(description="Train an optimized DQN agent on the BoxMoveEnv")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total timesteps to train")
    parser.add_argument("--horizon", type=int, default=50, help="Episode horizon")
    parser.add_argument("--n_boxes", type=int, default=5, help="Number of boxes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--train_freq", type=int, default=4, help="Train frequency")
    parser.add_argument("--target_update", type=int, default=1000, help="Target network update interval")
    parser.add_argument("--soft_update", action="store_true", help="Use soft target updates")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient (if enabled)")
    parser.add_argument("--exploration_fraction", type=float, default=0.1, help="Fraction of total timesteps for epsilon decay")
    parser.add_argument("--exploration_initial_eps", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--exploration_final_eps", type=float, default=0.05, help="Final exploration rate")
    parser.add_argument("--env_pool_size", type=int, default=8, help="Environment pool size")
    parser.add_argument("--eval_freq", type=int, default=5000, help="Evaluation frequency")
    parser.add_argument("--log_freq", type=int, default=1000, help="Logging frequency")
    parser.add_argument("--output_dir", type=str, default="optimized_results", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--resume", type=str, default=None, help="Path to model to resume training")
    
    return parser.parse_args()

def print_system_info():
    """Print system information for debugging."""
    print("\n--- System Information ---")
    print(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    mem = psutil.virtual_memory()
    print(f"Memory: {mem.total / (1024**3):.2f} GB total, {mem.available / (1024**3):.2f} GB available")
    if torch.cuda.is_available():
        print(f"CUDA available: Yes, device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
    else:
        print("CUDA available: No")
    print(f"PyTorch version: {torch.__version__}")
    print("---------------------\n")

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print_system_info()
    
    print("Training an optimized DQN agent with the following parameters:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    env = BoxMoveEnvGym(horizon=args.horizon, n_boxes=args.n_boxes, seed=args.seed)
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    tau = args.tau if args.soft_update else 1.0
    dqn = OptimizedDQN(
        env=env,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        learning_starts=1000,  # Start training after 1000 steps
        batch_size=args.batch_size,
        tau=tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        target_update_interval=args.target_update,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        seed=args.seed,
        device=device,
        env_pool_size=args.env_pool_size
    )
    
    if args.resume:
        print(f"Resuming training from {args.resume}")
        dqn.load(args.resume)
    
    # --- Start profiling ---
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    try:
        dqn.learn(
            total_timesteps=args.timesteps,
            eval_freq=args.eval_freq,
            log_freq=args.log_freq
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    profiler.disable()
    elapsed_time = time.time() - start_time
    
    print(f"Training took: {elapsed_time:.2f} seconds")
    
    # Print profiling results (top 20 functions by cumulative time)
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(20)
    # --- End profiling ---
    
    model_path = os.path.join(args.output_dir, "optimized_dqn_final.pt")
    dqn.save(model_path)
    dqn.plot_learning_curve(save_path=os.path.join(args.output_dir, "learning_curve.png"))
    
    print("\nRunning final evaluation...")
    mean_reward, mean_length = dqn.evaluate(num_episodes=20)
    
    with open(os.path.join(args.output_dir, "evaluation_results.txt"), 'w') as f:
        f.write("Final evaluation results:\n")
        f.write(f"Mean reward: {mean_reward:.2f}\n")
        f.write(f"Mean episode length: {mean_length:.2f}\n\n")
        f.write("Training parameters:\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.write(f"\nTraining time: {elapsed_time:.2f} seconds\n")
    
    print(f"All results saved to {args.output_dir}")

def quick_test():
    """Run a quick test to validate the learning improvements"""
    print("Running quick test to validate learning improvements...")
    
    # Set up simplified environment with fewer boxes
    env = BoxMoveEnvGym(horizon=20, n_boxes=2, seed=42)
    
    # Set up the DQN with optimized parameters
    dqn = OptimizedDQN(
        env=env,
        learning_rate=5e-4,
        buffer_size=10000,
        learning_starts=200,  # Start training after fewer steps
        batch_size=32,
        tau=0.1,  # More aggressive target updates
        gamma=0.99,
        train_freq=2,  # Train more frequently
        target_update_interval=100,
        exploration_fraction=0.5,  # More exploration
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        env_pool_size=4  # Smaller pool for quick test
    )
    
    # Run short training session
    dqn.learn(total_timesteps=1000, eval_freq=500, log_freq=100)
    
    # Save results
    dqn.save("quick_test_model.pt")
    dqn.plot_learning_curve(save_path="quick_test_learning_curve.png")
    
    # Print final summary
    print("\nQuick test complete!")
    print("Check quick_test_learning_curve.png to see the learning progress")

def benchmark_against_random():
    """Compare optimized DQN against a random policy in identical environments"""
    print("\n======== Benchmarking DQN vs Random Policy ========")
    
    # Configuration - increase complexity for better differentiation
    n_boxes = 8  # More boxes = more complex environment
    horizon = 50  # Longer horizon allows for more strategic moves
    n_episodes = 100  # More episodes for statistical significance
    benchmark_seeds = list(range(1000, 1000 + n_episodes))  # Fixed seeds for reproducibility
    
    # Initialize environment
    env = BoxMoveEnvGym(horizon=horizon, n_boxes=n_boxes, seed=42)
    
    # Set up the DQN with optimized parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load our pre-trained model
    model_path = "optimized_results/best_model.pt"
    if not os.path.exists(model_path):
        model_path = "quick_test_model.pt"  # Fallback to quick test model
    
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}. Training a new one...")
        dqn = OptimizedDQN(
            env=env,
            learning_rate=5e-4,
            buffer_size=10000,
            learning_starts=200,
            batch_size=32,
            tau=0.1,
            gamma=0.99,
            train_freq=2,
            target_update_interval=100,
            exploration_fraction=0.5,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,
            seed=42,
            device=device,
            env_pool_size=4
        )
        dqn.learn(total_timesteps=3000, eval_freq=1000, log_freq=250)
        dqn.save("benchmark_model.pt")
        model_path = "benchmark_model.pt"
    
    print(f"Loading pre-trained model from {model_path}")
    dqn = OptimizedDQN(
        env=env,
        learning_rate=5e-4,
        buffer_size=10000,
        learning_starts=200,
        batch_size=32,
        tau=0.1,
        gamma=0.99,
        train_freq=2,
        target_update_interval=100,
        exploration_fraction=0.0,  # No exploration during benchmark
        exploration_initial_eps=0.0,
        exploration_final_eps=0.0,
        seed=42,
        device=device,
        env_pool_size=4
    )
    dqn.load(model_path)
    
    # Warm-up period to ensure model is ready
    print("Performing warm-up runs...")
    for _ in range(10):
        warmup_seed = random.randint(0, 999)
        env.seed = warmup_seed
        obs = env.reset(seed=warmup_seed)
        done = False
        while not done:
            action, _ = dqn._select_action(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            obs = next_obs
    
    # Benchmark results
    dqn_rewards = []
    dqn_lengths = []
    dqn_success_count = 0
    dqn_choice_time = []
    
    random_rewards = []
    random_lengths = []
    random_success_count = 0
    random_choice_time = []
    
    print(f"\nRunning benchmark over {n_episodes} episodes with {n_boxes} boxes and horizon {horizon}...")
    for episode, seed in enumerate(benchmark_seeds):
        print(f"Episode {episode+1}/{n_episodes}", end="\r")
        
        # Run DQN policy on this seed
        env.seed = seed
        obs = env.reset(seed=seed)
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < horizon:
            # Time the decision-making process
            start_time = time.time()
            action, _ = dqn._select_action(obs, deterministic=True)
            end_time = time.time()
            dqn_choice_time.append(end_time - start_time)
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            obs = next_obs
        
        dqn_rewards.append(episode_reward)
        dqn_lengths.append(episode_length)
        if episode_reward > 0:  # Count successful episodes (positive reward)
            dqn_success_count += 1
        
        # Run random policy on the exact same seed
        env.seed = seed
        obs = env.reset(seed=seed)
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < horizon:
            # Time the decision-making process
            start_time = time.time()
            action_list = env.actions()
            valid_actions = set()
            valid_actions.update([a.id for a in action_list if hasattr(a, 'id')])
            action = random.choice(action_list) if action_list else None
            
            if action is None:
                # No valid actions, end episode
                truncated = True
                continue
            
            end_time = time.time()
            random_choice_time.append(end_time - start_time)
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            obs = next_obs
        
        random_rewards.append(episode_reward)
        random_lengths.append(episode_length)
        if episode_reward > 0:  # Count successful episodes (positive reward)
            random_success_count += 1
    
    # Compute statistics
    dqn_mean_reward = sum(dqn_rewards) / len(dqn_rewards)
    dqn_mean_length = sum(dqn_lengths) / len(dqn_lengths)
    dqn_success_rate = dqn_success_count / n_episodes * 100
    dqn_mean_choice_time = sum(dqn_choice_time) / len(dqn_choice_time) * 1000  # Convert to ms
    
    random_mean_reward = sum(random_rewards) / len(random_rewards)
    random_mean_length = sum(random_lengths) / len(random_lengths)
    random_success_rate = random_success_count / n_episodes * 100
    random_mean_choice_time = sum(random_choice_time) / len(random_choice_time) * 1000  # Convert to ms
    
    # Print results
    print("\n\n========== Benchmark Results ==========")
    print(f"Episodes: {n_episodes}")
    print(f"Environment: {n_boxes} boxes, {horizon} horizon")
    print("\nOptimized DQN:")
    print(f"  Mean reward: {dqn_mean_reward:.2f}")
    print(f"  Mean episode length: {dqn_mean_length:.2f}")
    print(f"  Success rate: {dqn_success_rate:.2f}%")
    print(f"  Mean decision time: {dqn_mean_choice_time:.4f} ms")
    print("\nRandom Policy:")
    print(f"  Mean reward: {random_mean_reward:.2f}")
    print(f"  Mean episode length: {random_mean_length:.2f}")
    print(f"  Success rate: {random_success_rate:.2f}%")
    print(f"  Mean decision time: {random_mean_choice_time:.4f} ms")
    print("\nPerformance Comparison:")
    reward_improvement = (dqn_mean_reward - random_mean_reward)
    if random_mean_reward > 0:
        reward_pct = ((dqn_mean_reward / random_mean_reward) - 1) * 100
    else:
        reward_pct = float('inf') if dqn_mean_reward > 0 else 0
    print(f"  Reward improvement: {reward_improvement:.2f} ({reward_pct:.2f}%)")
    print(f"  Success rate improvement: {(dqn_success_rate - random_success_rate):.2f} percentage points")
    
    # Plot results
    plt.figure(figsize=(14, 12))
    
    # Plot episode rewards
    plt.subplot(2, 2, 1)
    plt.plot(dqn_rewards, 'b-', alpha=0.5, label='DQN Policy')
    plt.plot(random_rewards, 'r-', alpha=0.5, label='Random Policy')
    
    # Add trend lines
    z = np.polyfit(range(len(dqn_rewards)), dqn_rewards, 1)
    p = np.poly1d(z)
    plt.plot(range(len(dqn_rewards)), p(range(len(dqn_rewards))), 'b--', linewidth=2)
    
    z = np.polyfit(range(len(random_rewards)), random_rewards, 1)
    p = np.poly1d(z)
    plt.plot(range(len(random_rewards)), p(range(len(random_rewards))), 'r--', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('DQN vs Random Policy: Episode Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot episode lengths
    plt.subplot(2, 2, 2)
    plt.plot(dqn_lengths, 'b-', alpha=0.5, label='DQN Policy')
    plt.plot(random_lengths, 'r-', alpha=0.5, label='Random Policy')
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('DQN vs Random Policy: Episode Lengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot histogram of rewards
    plt.subplot(2, 2, 3)
    plt.hist(dqn_rewards, bins=20, alpha=0.5, label='DQN Policy', color='blue')
    plt.hist(random_rewards, bins=20, alpha=0.5, label='Random Policy', color='red')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of Episode Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot success rate comparison
    plt.subplot(2, 2, 4)
    plt.bar(['DQN Policy', 'Random Policy'], [dqn_success_rate, random_success_rate], color=['blue', 'red'], alpha=0.7)
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.close()
    
    print("\nBenchmark results saved to benchmark_results.png")

def advanced_training():
    """Train an advanced DQN with optimized hyperparameters"""
    print("Running advanced training with optimized hyperparameters...")
    
    # Define advanced hyperparameters
    args = {
        "n_boxes": 6,                  # Moderate complexity
        "horizon": 30,                 # Reasonable episode length
        "timesteps": 15000,            # Reduced for testing (original: 50000)
        "buffer_size": 100000,         # Large replay buffer
        "batch_size": 64,              # Larger batch size for better gradient estimation
        "learning_rate": 5e-4,         # Slightly higher learning rate
        "gamma": 0.99,                 # Standard discount factor
        "train_freq": 2,               # Train more frequently
        "target_update": 1000,         # Update target network less frequently for stability
        "tau": 0.005,                  # Soft update coefficient
        "exploration_fraction": 0.3,   # Longer exploration phase
        "exploration_initial_eps": 1.0, # Start with full exploration
        "exploration_final_eps": 0.01, # End with minimal exploration
        "use_prioritized_replay": True, # Use prioritized experience replay
        "use_dueling_network": True,   # Use dueling network architecture
        "use_reward_shaping": True,    # Use reward shaping
        "double_q": True,              # Use double Q-learning
        "log_freq": 250,               # Log more frequently
        "eval_freq": 2500,             # Evaluate periodically
        "env_pool_size": 8,            # Reasonable environment pool size
        "alpha": 0.6,                  # Priority exponent
        "beta_start": 0.4,             # Initial importance sampling weight
        "seed": 42                     # Fixed seed for reproducibility
    }
    
    # Create output directory
    os.makedirs("advanced_training", exist_ok=True)
    
    print("Training with the following hyperparameters:")
    for key, value in args.items():
        print(f"  {key}: {value}")
    
    # Set up environment
    env = BoxMoveEnvGym(horizon=args["horizon"], n_boxes=args["n_boxes"], seed=args["seed"])
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create agent with all improvements
    dqn = OptimizedDQN(
        env=env,
        learning_rate=args["learning_rate"],
        buffer_size=args["buffer_size"],
        learning_starts=1000,  # Start training after 1000 steps
        batch_size=args["batch_size"],
        tau=args["tau"],
        gamma=args["gamma"],
        train_freq=args["train_freq"],
        target_update_interval=args["target_update"],
        exploration_fraction=args["exploration_fraction"],
        exploration_initial_eps=args["exploration_initial_eps"],
        exploration_final_eps=args["exploration_final_eps"],
        use_prioritized_replay=args["use_prioritized_replay"],
        use_dueling_network=args["use_dueling_network"],
        use_reward_shaping=args["use_reward_shaping"],
        double_q=args["double_q"],
        alpha=args["alpha"],
        beta_start=args["beta_start"],
        seed=args["seed"],
        device=device,
        env_pool_size=args["env_pool_size"]
    )
    
    # Run training
    start_time = time.time()
    dqn.learn(
        total_timesteps=args["timesteps"],
        eval_freq=args["eval_freq"],
        log_freq=args["log_freq"]
    )
    
    # Calculate training time and save final model
    training_time = time.time() - start_time
    model_path = os.path.join("advanced_training", "final_model.pt")
    dqn.save(model_path)
    
    # Save learning curve
    dqn.plot_learning_curve(save_path=os.path.join("advanced_training", "learning_curve.png"))
    
    # Run final evaluation
    print("\nRunning final evaluation...")
    mean_reward, mean_length = dqn.evaluate(num_episodes=20)
    
    # Save evaluation results
    with open(os.path.join("advanced_training", "results.txt"), 'w') as f:
        f.write("Advanced DQN Training Results\n")
        f.write("=============================\n\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write(f"Total timesteps: {args['timesteps']}\n")
        f.write(f"Final evaluation mean reward: {mean_reward:.2f}\n")
        f.write(f"Final evaluation mean episode length: {mean_length:.2f}\n\n")
        f.write("Hyperparameters:\n")
        for key, value in args.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"\nAdvanced training complete! Results saved to advanced_training/")
    print(f"Final evaluation mean reward: {mean_reward:.2f}")
    
    return dqn

def compare_agents():
    """Simplified comparison between random policy and DQN"""
    print("Running simplified comparison between random policy and DQN...")
    
    # Environment configuration 
    n_boxes = 4
    horizon = 20
    n_episodes = 20
    seed = 42
    
    # Create environment
    env = BoxMoveEnvGym(horizon=horizon, n_boxes=n_boxes, seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize simple DQN agent (already trained)
    dqn = OptimizedDQN(
        env,
        learning_rate=3e-4,
        buffer_size=10000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        device=device,
        seed=seed,
        env_pool_size=4,
        use_prioritized_replay=False,
        use_dueling_network=False,
        use_reward_shaping=False
    )
    
    # Try to load the standard model
    try:
        print("Loading DQN from standard_dqn.pt")
        dqn.load("standard_dqn.pt")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Could not load DQN model: {e}")
        print("Training DQN for 5000 timesteps...")
        dqn.learn(total_timesteps=5000, log_freq=1000)
        dqn.save("standard_dqn.pt")
    
    # Evaluate using DQN's built-in evaluation method
    print("\nEvaluating DQN policy...")
    mean_reward, mean_length = dqn.evaluate(num_episodes=n_episodes)
    
    # Create lists of rewards and lengths for plotting
    dqn_rewards = [mean_reward] * n_episodes  # Approximate since we only have the mean
    dqn_lengths = [mean_length] * n_episodes  # Approximate since we only have the mean
    
    # Manually evaluate to get individual episode results
    print("\nPerforming detailed DQN evaluation...")
    dqn_rewards = []
    dqn_lengths = []
    
    for i in range(n_episodes):
        print(f"Episode {i+1}/{n_episodes}", end="\r")
        obs = env.reset(seed=i+100)  # Same seeds used for random policy
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
    
    # Evaluate random policy
    print("\nEvaluating random policy...")
    random_rewards = []
    random_lengths = []
    
    for i in range(n_episodes):
        print(f"Episode {i+1}/{n_episodes}", end="\r")
        env.reset(seed=i+100)  # Different seed for fair comparison
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Random action selection
            valid_actions = env.actions()
            if not valid_actions:
                break
                
            # Select random action from valid actions
            action_idx = env.action_space.sample()
            
            # Take action in environment
            _, reward, terminated, truncated, _ = env.step(action_idx)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
        
        random_rewards.append(episode_reward)
        random_lengths.append(episode_length)
    
    # Calculate statistics
    random_mean_reward = np.mean(random_rewards)
    random_std_reward = np.std(random_rewards)
    random_mean_length = np.mean(random_lengths)
    random_success_rate = sum(1 for r in random_rewards if r > 0) / len(random_rewards) * 100
    
    dqn_mean_reward = np.mean(dqn_rewards)
    dqn_std_reward = np.std(dqn_rewards)
    dqn_mean_length = np.mean(dqn_lengths)
    dqn_success_rate = sum(1 for r in dqn_rewards if r > 0) / len(dqn_rewards) * 100
    
    # Print results
    print("\n--- Benchmark Results ---")
    print("\nRandom Policy:")
    print(f"  Mean Reward: {random_mean_reward:.2f} ± {random_std_reward:.2f}")
    print(f"  Mean Episode Length: {random_mean_length:.2f}")
    print(f"  Success Rate: {random_success_rate:.2f}%")
    
    print("\nDQN Policy:")
    print(f"  Mean Reward: {dqn_mean_reward:.2f} ± {dqn_std_reward:.2f}")
    print(f"  Mean Episode Length: {dqn_mean_length:.2f}")
    print(f"  Success Rate: {dqn_success_rate:.2f}%")
    
    print("\nPerformance Comparison:")
    improvement = (dqn_mean_reward - random_mean_reward) / abs(random_mean_reward) * 100 if random_mean_reward != 0 else 0
    print(f"  DQN vs Random: {improvement:.2f}% improvement in reward")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot episode rewards
    plt.subplot(2, 2, 1)
    plt.plot(random_rewards, label='Random', alpha=0.7)
    plt.plot(dqn_rewards, label='DQN', alpha=0.7)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot reward distributions
    plt.subplot(2, 2, 2)
    plt.hist(random_rewards, alpha=0.5, label='Random', bins=10)
    plt.hist(dqn_rewards, alpha=0.5, label='DQN', bins=10)
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot episode lengths
    plt.subplot(2, 2, 3)
    plt.plot(random_lengths, label='Random', alpha=0.7)
    plt.plot(dqn_lengths, label='DQN', alpha=0.7)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot mean metrics with error bars
    plt.subplot(2, 2, 4)
    x = ['Mean Reward', 'Mean Length']
    x_pos = np.arange(len(x))
    width = 0.35
    
    # Mean values with error bars
    plt.bar(x_pos - width/2, [random_mean_reward, random_mean_length], width, 
            label='Random', alpha=0.7, yerr=[random_std_reward, 0])
    plt.bar(x_pos + width/2, [dqn_mean_reward, dqn_mean_length], width, 
            label='DQN', alpha=0.7, yerr=[dqn_std_reward, 0])
    
    plt.title('Key Metrics')
    plt.xticks(x_pos, x)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('agent_comparison.png')
    plt.close()
    print("Results visualization saved to agent_comparison.png")
    
    # Save detailed results to file
    with open("comparison_results.txt", "w") as f:
        f.write("=== Box Move Environment Agent Comparison ===\n\n")
        f.write(f"Configuration: {n_episodes} episodes, {n_boxes} boxes, horizon {horizon}\n\n")
        
        f.write("--- Random Policy ---\n")
        f.write(f"Mean Reward: {random_mean_reward:.2f} ± {random_std_reward:.2f}\n")
        f.write(f"Mean Episode Length: {random_mean_length:.2f}\n")
        f.write(f"Success Rate: {random_success_rate:.2f}%\n\n")
        
        f.write("--- DQN Policy ---\n")
        f.write(f"Mean Reward: {dqn_mean_reward:.2f} ± {dqn_std_reward:.2f}\n")
        f.write(f"Mean Episode Length: {dqn_mean_length:.2f}\n")
        f.write(f"Success Rate: {dqn_success_rate:.2f}%\n\n")
        
        f.write("--- Performance Comparison ---\n")
        f.write(f"DQN vs Random: {improvement:.2f}% improvement in reward\n")
    
    print("Detailed results saved to comparison_results.txt")
    return

if __name__ == "__main__":
    # Add more command-line options
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick-test":
            quick_test()
        elif sys.argv[1] == "--benchmark":
            benchmark_against_random()
        elif sys.argv[1] == "--advanced":
            advanced_training()
        elif sys.argv[1] == "--compare":
            compare_agents()
        else:
            main()
    else:
        main()
