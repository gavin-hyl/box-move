import numpy as np

from BoxMoveEnvGym import BoxMoveEnvGym
from Constants import DATA_DIR

def generate_training_data(num_episodes=50, max_steps=20):
    """
    Generate training samples by running the BoxMove environment.
    
    For each step, we extract a 3D state and action representation.
    The state is represented as two arrays (one for each zone) obtained from state_3d(),
    and the action is represented as two arrays from the chosen action.
    
    Returns:
        data: List of tuples (state_zone0, state_zone1, action_zone0, action_zone1, reward)
    """
    data = []
    env = BoxMoveEnvGym(horizon=50, n_boxes=15)
    
    for ep in range(num_episodes):
        env.reset()
        done = False
        truncated = False
        steps = 0
        episode_data = []
        episode_reward = 0
        
        while not done and not truncated and steps < max_steps:
            # Get current 3D state (list: [zone0_dense, zone1_dense])
            state_3d = env.env.state_3d()  # using the underlying environment
            
            # Retrieve valid actions.
            valid_actions = env.env.actions()
            if len(valid_actions) == 0:
                break
            
            # Randomly choose one valid action.
            chosen_action = np.random.choice(valid_actions)
            
            # Get the 3D representation for the action.
            action_3d = env.env.action_3d(chosen_action)
            
            # Find the discrete action index corresponding to chosen_action.
            action_idx = None
            for idx, act in env._action_map.items():
                if act == chosen_action:
                    action_idx = idx
                    break
            if action_idx is None:
                steps += 1
                continue
            
            # Take the step.
            next_state, reward, done, truncated, info = env.step(action_idx)
            
            # Append the training sample:
            # (state_zone0, state_zone1, action_zone0, action_zone1, reward)
            episode_data.append((state_3d[0].copy(), state_3d[1].copy(),
                                 action_3d[0].copy(), action_3d[1].copy()))
            episode_reward = reward
            steps += 1
        
        for d in episode_data:
            data.append((d[0], d[1], d[2], d[3], episode_reward))
    
    # Save training data.
    return data


def save_data(data, filename):
    """
    Save the training data to a file.
    
    Args:
        data: List of tuples (state_zone0, state_zone1, action_zone0, action_zone1, reward)
        filename: Name of the file to save the data.
    """
    # states and actions sep
    state0 = [d[0] for d in data]
    state1 = [d[1] for d in data]
    action0 = [d[2] for d in data]
    action1 = [d[3] for d in data]
    rewards = [d[4] for d in data]
    np.save(f"{DATA_DIR}/{filename}_states0.npy", state0)
    np.save(f"{DATA_DIR}/{filename}_states1.npy", state1)
    np.save(f"{DATA_DIR}/{filename}_actions0.npy", action0)
    np.save(f"{DATA_DIR}/{filename}_actions1.npy", action1)
    np.save(f"{DATA_DIR}/{filename}_rewards.npy", rewards)
    print(f"Data saved to '{filename}'.")


def load_data(filename):
    """
    Load the training data from a file.
    
    Args:
        filename: Name of the file to load the data from.
    
    Returns:
        data: List of tuples (state_zone0, state_zone1, action_zone0, action_zone1, reward)
    """
    state0 = np.load(f"{DATA_DIR}/{filename}_states0.npy", allow_pickle=True)
    state1 = np.load(f"{DATA_DIR}/{filename}_states1.npy", allow_pickle=True)
    action0 = np.load(f"{DATA_DIR}/{filename}_actions0.npy", allow_pickle=True)
    action1 = np.load(f"{DATA_DIR}/{filename}_actions1.npy", allow_pickle=True)
    rewards = np.load(f"{DATA_DIR}/{filename}_rewards.npy", allow_pickle=True)
    
    data = [(state0[i], state1[i], action0[i], action1[i], rewards[i]) for i in range(len(state0))]
    return data


if __name__ == "__main__":
    N_EP = 100
    training_data = generate_training_data(num_episodes=N_EP, max_steps=20)
    save_data(training_data, f"training_data{N_EP}")