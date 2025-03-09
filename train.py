#!/usr/bin/env python3
"""
Main entry point for training and evaluating the Box Move DQN agent.
"""
import os
import sys
import argparse
import torch

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.BoxMoveEnvGym import BoxMoveEnvGym
from src.OptimizedDQN import OptimizedDQN

def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate a DQN agent on the BoxMoveEnv")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "benchmark"], default="train",
                       help="Mode to run in: train, evaluate, or benchmark")
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
    parser.add_argument("--model_path", type=str, default="models/dqn_model.pt", help="Path to save/load model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Initialize environment
    env = BoxMoveEnvGym(
        horizon=args.horizon,
        n_boxes=args.n_boxes,
        seed=args.seed
    )
    
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
    
    if args.mode == "train":
        print(f"Training DQN for {args.timesteps} timesteps...")
        dqn.learn(
            total_timesteps=args.timesteps,
            eval_freq=args.eval_freq,
            log_freq=args.log_freq
        )
        # Save model and learning curve
        dqn.save(args.model_path)
        dqn.plot_learning_curve(save_path="results/learning_curve.png")
        print(f"Training complete! Model saved to {args.model_path}")
    
    elif args.mode == "evaluate":
        # Try to load a pre-trained model
        try:
            dqn.load(args.model_path)
            print(f"Model loaded from {args.model_path}")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Training a new model for evaluation...")
            dqn.learn(total_timesteps=5000, log_freq=500)
        
        # Evaluate the agent
        print("\nEvaluating DQN performance...")
        mean_reward, mean_length = dqn.evaluate(num_episodes=10)
        print(f"Mean reward: {mean_reward:.2f}, Mean episode length: {mean_length:.2f}")
    
    elif args.mode == "benchmark":
        from src.Benchmark import evaluate_policy, plot_comparison
        from src.Benchmark import RandomPolicy, GreedyPolicy, DQNPolicy
        
        # Try to load a pre-trained model
        try:
            dqn.load(args.model_path)
            print(f"Model loaded from {args.model_path}")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Training a new model for benchmarking...")
            dqn.learn(total_timesteps=5000, log_freq=500)
            dqn.save(args.model_path)
        
        # Resolve device
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create policies
        random_policy = RandomPolicy(env)
        greedy_policy = GreedyPolicy(env)
        dqn_policy = DQNPolicy(env, args.model_path, device=device)
        
        # Evaluate each policy
        print("\nEvaluating policies...")
        metrics = {}
        
        print("Evaluating Random Policy...")
        metrics["Random"] = evaluate_policy(env, random_policy, num_episodes=50, seed=args.seed)
        
        print("\nEvaluating Greedy Policy...")
        metrics["Greedy"] = evaluate_policy(env, greedy_policy, num_episodes=50, seed=args.seed)
        
        print("\nEvaluating DQN Policy...")
        metrics["DQN"] = evaluate_policy(env, dqn_policy, num_episodes=50, seed=args.seed)
        
        # Plot comparison
        plot_comparison(metrics, save_path="results/benchmark_comparison.png")
        print("\nBenchmark complete! Results saved to results/benchmark_comparison.png")

if __name__ == "__main__":
    main() 