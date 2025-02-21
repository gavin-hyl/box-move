from BoxMoveEnv import BoxMoveEnv
from BoxMoveEnvGym import BoxMoveEnvGym
import random
import Box

def box_test():
    print("===Box Test===")
    b = Box.make([0, 0, 0], [1, 2, 3], 0)
    print(b)
    print("pos:", Box.pos(b))
    print("size:", Box.size(b))
    print("zone:", Box.zone(b))
    print("bottom_face:", Box.bottom_face(b))
    print("top_face:", Box.top_face(b))

def bme_test():
    print("===BoxMoveEnv Test===")
    bme = BoxMoveEnv(n_boxes=20)
    bme.reset()
    print("Zone 1 top:", bme.zone1_top)
    print("Boxes:", Box.boxes_from_state(bme.state))

    bme.visualize_scene()
    actions = bme.actions()
    while actions:
        action_idx = random.randint(0, len(actions) - 1)
        bme.step(actions[action_idx])
        bme.visualize_scene()
        actions = bme.actions()
    print("no more actions")


def gym_test():
    # Create the environment with a short horizon and a few boxes for testing.
    env = BoxMoveEnvGym(horizon=10, gamma=1, n_boxes=3)
    
    # Reset the environment
    obs = env.reset(n_boxes=3)
    print("Initial observation:\n", obs)
    
    # Check that the observation is contained in the observation space.
    assert env.observation_space.contains(obs), "Initial observation is not within the observation space."
    
    # Get the list of valid actions from the underlying environment.
    valid_box_actions = env.env.actions()
    print("Valid actions from the environment:", len(valid_box_actions))
    
    # Find a discrete action index that corresponds to a valid action.
    valid_action_index = None
    for idx, discrete_action in env._action_map.items():
        for valid_action in valid_box_actions:
            if discrete_action == valid_action:
                valid_action_index = idx
                break
        if valid_action_index is not None:
            break
    
    if valid_action_index is None:
        print("No valid action found. This might indicate the environment is in a terminal state.")
    else:
        print("Taking valid action index:", valid_action_index)
        obs, reward, done, info = env.step(valid_action_index)
        print("After valid action:")
        print("Observation:\n", obs)
        print("Reward:", reward)
        print("Done:", done)
        print("Info:", info)
    
    # Now, try to find a discrete action index that is not valid.
    invalid_action_index = None
    for idx, discrete_action in env._action_map.items():
        is_valid = any(
            discrete_action == va
            for va in valid_box_actions
        )
        if not is_valid:
            invalid_action_index = idx
            break
    
    if invalid_action_index is not None:
        print("Taking invalid action index:", invalid_action_index)
        obs, reward, done, info = env.step(invalid_action_index)
        print("After invalid action:")
        print("Observation:\n", obs)
        print("Reward (expected penalty -1):", reward)
        print("Done:", done)
        print("Info (should indicate invalid action):", info)
    else:
        print("Could not find an invalid action. (This is unlikely if the state is non-terminal.)")
    
    # Test the render functionality.
    try:
        env.render()
        print("Render executed successfully.")
    except Exception as e:
        print("Render failed with exception:", e)


if __name__ == "__main__":
    # box_test()
    bme_test()
    # gym_test()