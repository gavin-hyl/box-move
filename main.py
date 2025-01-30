import numpy as np
from BoxEnv import BoxMoveEnvGym
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy

def main():
    # Example zone sizes
    zone_sizes = [
        np.array([5, 5, 5], dtype=int),  # zone 0
        np.array([5, 5, 5], dtype=int)   # zone 1
    ]

    # Create the Gym environment
    env = BoxMoveEnvGym(zone_sizes=zone_sizes, horizon=20, max_boxes=5)

    # Create a DQN model (with default MLP policy)
    model = DQN(
        policy=MlpPolicy,
        env=env,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        verbose=1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        exploration_fraction=0.2,
        target_update_interval=500
    )

    # Train the model
    model.learn(total_timesteps=50000)

    # Test the trained agent
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        print(f"Action: {action}, Reward: {reward}, Done: {done}")

if __name__ == "__main__":
    main()
