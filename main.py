import numpy as np
from BoxGymEnv import BoxMoveEnvGym
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from Core import _is_valid_action, all_actions
import Dense

def main():
    env = BoxMoveEnvGym(horizon=20)

    model = DQN(
        policy=MlpPolicy,
        env=env,
        learning_rate=1e-3,
        buffer_size=1000,
        learning_starts=100,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        verbose=1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        exploration_fraction=0.2,
        target_update_interval=50
    )

    model.learn(total_timesteps=1000)

    # Test the trained agent
    obs = env.reset()[0]
    done = False
    truncated = False
    while not done and not truncated:
        action, _states = model.predict(obs, deterministic=True)
        all_action_list = all_actions()
        if not _is_valid_action(env.state, all_action_list[action]):
            print("Invalid action")
            break
        print(f"Action: {Dense.dense_action(obs, all_action_list[action])}")
        print(f"State: {Dense.dense_state(env.state)}")
        obs, reward, done, truncated, info = env.step(action)
    print(f"Dense state: {Dense.dense_state(env.state)}")

if __name__ == "__main__":
    main()
