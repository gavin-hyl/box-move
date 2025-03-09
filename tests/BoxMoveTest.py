"""
Comprehensive test suite for the combined BoxMoveEnvGym implementation.
Tests initialization, state management, actions, rewards, and edge cases.
"""
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Box import Box
from src.BoxAction import BoxAction
from src.BoxMoveEnvGym import BoxMoveEnvGym
from src.Constants import ZONE0, ZONE1

def test_initialization():
    """Test environment initialization with different parameters."""
    print("\n===== Testing Initialization =====")
    
    # Test default initialization
    env = BoxMoveEnvGym()
    print(f"Default initialization - n_boxes: {len(env.boxes)}, horizon: {env.horizon}, gamma: {env.gamma}")
    
    # Test with custom parameters
    env = BoxMoveEnvGym(horizon=50, gamma=0.9, n_boxes=10, seed=42)
    print(f"Custom initialization - n_boxes: {len(env.boxes)}, horizon: {env.horizon}, gamma: {env.gamma}")
    
    # Test with zero boxes
    env = BoxMoveEnvGym(n_boxes=0)
    print(f"Zero boxes - n_boxes: {len(env.boxes)}")
    
    # Test with max boxes (should be limited by zone size)
    max_boxes = np.prod(ZONE0)
    env = BoxMoveEnvGym(n_boxes=int(max_boxes))
    print(f"Max boxes - requested: {max_boxes}, actual: {len(env.boxes)}")
    
    # Test seeded initialization (should be deterministic)
    env1 = BoxMoveEnvGym(n_boxes=5, seed=123)
    env2 = BoxMoveEnvGym(n_boxes=5, seed=123)
    boxes_match = all(
        np.array_equal(b1.pos, b2.pos) and 
        np.array_equal(b1.size, b2.size) and 
        b1.zone == b2.zone and 
        b1.val == b2.val
        for b1, b2 in zip(env1.boxes, env2.boxes)
    )
    print(f"Seeded initialization - boxes match: {boxes_match}")
    
    return env  # Return an environment for use in other tests

def test_reset():
    """Test environment reset functionality."""
    print("\n===== Testing Reset =====")
    
    env = BoxMoveEnvGym(n_boxes=5)
    print(f"Initial state - n_boxes: {len(env.boxes)}")
    
    # Test reset with same number of boxes
    obs = env.reset()
    print(f"Reset with default boxes - n_boxes: {len(env.boxes)}")
    print(f"Observation shapes - zone0: {obs['state_zone0'].shape}, zone1: {obs['state_zone1'].shape}")
    
    # Test reset with different number of boxes
    obs = env.reset(n_boxes=10)
    print(f"Reset with 10 boxes - n_boxes: {len(env.boxes)}")
    
    # Test reset with zero boxes
    obs = env.reset(n_boxes=0)
    print(f"Reset with 0 boxes - n_boxes: {len(env.boxes)}")
    
    return env

def test_actions():
    """Test action generation and validity."""
    print("\n===== Testing Actions =====")
    
    env = BoxMoveEnvGym(n_boxes=10, seed=42)
    valid_actions = env.actions()
    print(f"Number of valid actions: {len(valid_actions)}")
    
    if valid_actions:
        # Test action equality
        action = valid_actions[0]
        same_action = BoxAction(action.pos_from, action.pos_to)
        diff_action = BoxAction(action.pos_from, (0, 0, 0))
        print(f"Action equality test - same: {action == same_action}, different: {action == diff_action}")
        
        # Test finding discrete action index
        action_idx = None
        for idx, act in env._action_map.items():
            if act == action:
                action_idx = idx
                break
        print(f"Found discrete index for action: {action_idx is not None}")
        
        # Test action after step
        if action_idx is not None:
            env.step(action_idx)
            new_valid_actions = env.actions()
            print(f"Valid actions after step: {len(new_valid_actions)}")
    else:
        print("No valid actions in initial state.")
    
    # Test action generation with empty zones
    env.reset(n_boxes=0)
    empty_actions = env.actions()
    print(f"Valid actions with no boxes: {len(empty_actions)}")
    
    return env

def test_step():
    """Test step function with valid and invalid actions."""
    print("\n===== Testing Step =====")
    
    env = BoxMoveEnvGym(n_boxes=5, seed=42)
    
    # Get valid actions
    valid_actions = env.actions()
    
    if not valid_actions:
        print("No valid actions available. Skipping step tests.")
        return env
    
    # Test with valid action
    valid_action = valid_actions[0]
    valid_idx = None
    for idx, act in env._action_map.items():
        if act == valid_action:
            valid_idx = idx
            break
    
    if valid_idx is not None:
        print(f"Testing valid action index: {valid_idx}")
        obs, reward, done, truncated, info = env.step(valid_idx)
        print(f"Valid step result - reward: {reward}, done: {done}, truncated: {truncated}")
        print(f"Observation shapes - zone0: {obs['state_zone0'].shape}, zone1: {obs['state_zone1'].shape}")
    
    # Test with invalid action
    invalid_idx = -1
    for idx in env._action_map.keys():
        action = env._action_map[idx]
        is_valid = any(action == va for va in valid_actions)
        if not is_valid:
            invalid_idx = idx
            break
    
    if invalid_idx != -1:
        print(f"Testing invalid action index: {invalid_idx}")
        obs, reward, done, truncated, info = env.step(invalid_idx)
        print(f"Invalid step result - reward: {reward}, done: {done}, truncated: {truncated}")
        print(f"Info: {info}")
    
    # Test step until done
    env.reset(n_boxes=3)
    steps = 0
    done = False
    truncated = False
    total_reward = 0
    
    while not (done or truncated) and steps < 100:
        actions = env.actions()
        if not actions:
            break
            
        action = random.choice(actions)
        action_idx = None
        for idx, act in env._action_map.items():
            if act == action:
                action_idx = idx
                break
                
        if action_idx is None:
            break
            
        _, reward, done, truncated, _ = env.step(action_idx)
        total_reward += reward
        steps += 1
    
    print(f"Episode completed - steps: {steps}, total reward: {total_reward}")
    
    return env

def test_reward():
    """Test reward calculation."""
    print("\n===== Testing Reward =====")
    
    env = BoxMoveEnvGym(n_boxes=5, seed=42)
    
    # Test reward at initialization (should be 0 as not terminal)
    initial_reward = env.reward()
    print(f"Initial reward: {initial_reward}")
    
    # Play episode and check final reward
    steps = 0
    done = False
    truncated = False
    
    while not (done or truncated) and steps < 20:
        actions = env.actions()
        if not actions:
            break
            
        action = random.choice(actions)
        action_idx = None
        for idx, act in env._action_map.items():
            if act == action:
                action_idx = idx
                break
                
        if action_idx is None:
            break
            
        _, reward, done, truncated, _ = env.step(action_idx)
        steps += 1
    
    # Calculate expected reward (sum of values in zone1)
    zone1_values = sum(box.val for box in env.boxes if box.zone == 1)
    print(f"Final reward: {reward}, Zone1 sum: {zone1_values}")
    
    # Test occupancy
    occupancy = env.occupancy()
    print(f"Zone1 occupancy: {occupancy:.2f}")
    
    return env

def test_state_representation():
    """Test state representation methods."""
    print("\n===== Testing State Representation =====")
    
    env = BoxMoveEnvGym(n_boxes=5, seed=42)
    
    # Test 1D state
    state_1d = env.state_1d()
    print(f"1D state length: {len(state_1d)}")
    
    # Test 3D state
    state_3d = env.state_3d()
    print(f"3D state shapes - zone0: {state_3d[0].shape}, zone1: {state_3d[1].shape}")
    
    # Check 3D state sum (should match box values)
    zone0_sum = np.sum(state_3d[0])
    zone1_sum = np.sum(state_3d[1])
    expected_zone0 = sum(box.val_density() * np.prod(box.size) for box in env.boxes if box.zone == 0)
    expected_zone1 = sum(box.val_density() * np.prod(box.size) for box in env.boxes if box.zone == 1)
    
    print(f"Zone0 sum: {zone0_sum:.2f}, Expected: {expected_zone0:.2f}")
    print(f"Zone1 sum: {zone1_sum:.2f}, Expected: {expected_zone1:.2f}")
    
    return env

def test_action_representation():
    """Test action representation methods."""
    print("\n===== Testing Action Representation =====")
    
    env = BoxMoveEnvGym(n_boxes=5, seed=42)
    valid_actions = env.actions()
    
    if not valid_actions:
        print("No valid actions available. Skipping action representation tests.")
        return env
    
    # Test action_1d
    action = valid_actions[0]
    action_1d = env.action_1d(action)
    print(f"1D action representation: {action_1d}")
    
    # Test action_3d
    action_3d = env.action_3d(action)
    print(f"3D action shapes - zone0: {action_3d[0].shape}, zone1: {action_3d[1].shape}")
    
    # Check if action representation properly shows source and destination
    source_pos = action.pos_from
    target_pos = action.pos_to
    
    has_source = action_3d[0][source_pos] != 0
    has_target = action_3d[1][target_pos] != 0
    
    print(f"Action representation - source marked: {has_source}, target marked: {has_target}")
    
    return env

def test_edge_cases():
    """Test edge cases and corner conditions."""
    print("\n===== Testing Edge Cases =====")
    
    # Test with zero boxes
    env = BoxMoveEnvGym(n_boxes=0)
    actions = env.actions()
    print(f"Zero boxes - valid actions: {len(actions)}")
    
    # Test with max timesteps
    env = BoxMoveEnvGym(n_boxes=2, horizon=1)
    env.t = 1  # Set current time to horizon
    done = (env.t == env.horizon)
    print(f"Max timesteps - done: {done}")
    
    # Test with full zone1
    env = BoxMoveEnvGym(n_boxes=0)
    
    # Create a box that fills the entire zone1
    full_box = Box(np.array([0, 0, 0]), np.array(ZONE1), 1, 10)
    env.boxes = [full_box]
    env.set_boxes(env.boxes)
    
    # Check that zone1_top is empty (no available spots)
    print(f"Full zone1 - available top positions: {len(env.zone1_top)}")
    
    return env

def test_visualization():
    """Test visualization functionality."""
    print("\n===== Testing Visualization =====")
    
    env = BoxMoveEnvGym(n_boxes=3, seed=42)
    
    try:
        # Turn off interactive mode and use a non-GUI backend
        plt.ioff()
        plt.switch_backend('agg')
        
        # Test visualize_scene
        env.visualize_scene()
        plt.close()
        print("Visualization successful")
        
        # Test render
        env.render()
        plt.close()
        print("Render successful")
        
    except Exception as e:
        print(f"Visualization error: {e}")
    
    return env

def test_performance():
    """Test performance metrics for the environment."""
    print("\n===== Testing Performance =====")
    
    # Test initialization time for different numbers of boxes
    box_counts = [5, 10, 20, 50]
    init_times = []
    
    for n_boxes in box_counts:
        start_time = time.time()
        env = BoxMoveEnvGym(n_boxes=n_boxes)
        init_time = time.time() - start_time
        init_times.append(init_time)
        print(f"Initialization with {n_boxes} boxes: {init_time:.4f} seconds")
    
    # Test step time
    env = BoxMoveEnvGym(n_boxes=10)
    valid_actions = env.actions()
    
    if valid_actions:
        action = random.choice(valid_actions)
        action_idx = None
        for idx, act in env._action_map.items():
            if act == action:
                action_idx = idx
                break
                
        if action_idx is not None:
            start_time = time.time()
            env.step(action_idx)
            step_time = time.time() - start_time
            print(f"Step time: {step_time:.4f} seconds")
    
    # Test actions generation time
    start_time = time.time()
    env.actions()
    actions_time = time.time() - start_time
    print(f"Actions generation time: {actions_time:.4f} seconds")
    
    return env

def run_episode(env, max_steps=100, render=False):
    """Run a complete episode and return total reward and steps."""
    env.reset()
    total_reward = 0
    steps = 0
    done = False
    truncated = False
    
    while not (done or truncated) and steps < max_steps:
        actions = env.actions()
        if not actions:
            break
            
        action = random.choice(actions)
        action_idx = None
        for idx, act in env._action_map.items():
            if act == action:
                action_idx = idx
                break
                
        if action_idx is None:
            break
            
        _, reward, done, truncated, _ = env.step(action_idx)
        total_reward += reward
        steps += 1
        
        if render:
            env.render()
    
    return total_reward, steps

def run_all_tests():
    """Run all the test functions."""
    print("\n===== STARTING COMPREHENSIVE TESTS =====")
    
    test_initialization()
    test_reset()
    test_actions()
    test_step()
    test_reward()
    test_state_representation()
    test_action_representation()
    test_edge_cases()
    test_visualization()
    test_performance()
    
    # Run complete episodes
    print("\n===== Running Complete Episodes =====")
    env = BoxMoveEnvGym(n_boxes=5, seed=42)
    
    num_episodes = 3
    rewards = []
    steps_list = []
    
    for i in range(num_episodes):
        reward, steps = run_episode(env)
        rewards.append(reward)
        steps_list.append(steps)
        print(f"Episode {i+1}: Reward = {reward}, Steps = {steps}")
    
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Average steps: {np.mean(steps_list):.2f}")
    
    print("\n===== ALL TESTS COMPLETED =====")

if __name__ == "__main__":
    run_all_tests()