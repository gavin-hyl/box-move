import os
import time
import argparse
import torch
import psutil
import cProfile
import pstats

from BoxMoveEnvGym import BoxMoveEnvGym
from optimized_dqn import OptimizedDQN  # Ensure this file is in the same directory

def parse_args():
    parser = argparse.ArgumentParser(description="Train an optimized DQN agent on the BoxMoveEnv")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total timesteps to train")
    parser.add_argument("--horizon", type=int, default=50, help="Episode horizon")
    parser.add_argument("--n_boxes", type=int, default=10, help="Number of boxes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--train_freq", type=int, default=4, help="Train frequency")
    parser.add_argument("--target_update", type=int, default=1000, help="Target network update interval")
    parser.add_argument("--soft_update", action="store_true", help="Use soft target updates")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient (if enabled)")
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

if __name__ == "__main__":
    main()
