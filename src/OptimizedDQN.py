import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import mse_loss
from tqdm import tqdm
import sys

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your environment and model components
from src.BoxMoveEnvGym import BoxMoveEnvGym
from src.QNet import CNNQNetwork

# Linear schedule for epsilon decay
class LinearSchedule:
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over schedule_timesteps.
        After this many timesteps pass final_p is returned.
        
        Args:
            schedule_timesteps: Number of timesteps for which to linearly anneal initial_p to final_p
            initial_p: Initial output value
            final_p: Final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

# Optimized replay buffer for dict observation spaces
class OptimizedReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device=torch.device("cpu")):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.pos = 0
        self.full = False
        
        self.observations = {}
        self.next_observations = {}
        for key, space in observation_space.spaces.items():
            self.observations[key] = np.zeros((buffer_size, *space.shape), dtype=np.float32)
            self.next_observations[key] = np.zeros((buffer_size, *space.shape), dtype=np.float32)
        
        self.actions = np.zeros((buffer_size, 1), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
    
    def add(self, obs, next_obs, action, reward, done, info):
        for key in self.observations.keys():
            self.observations[key][self.pos] = obs[key].astype(np.float32)
            self.next_observations[key][self.pos] = next_obs[key].astype(np.float32)
        
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
    
    def sample(self, batch_size):
        if self.full:
            indices = np.random.randint(0, self.buffer_size, size=batch_size)
        else:
            indices = np.random.randint(0, self.pos, size=batch_size)
        
        obs = {}
        next_obs = {}
        for key in self.observations.keys():
            obs[key] = torch.as_tensor(self.observations[key][indices], device=self.device)
            next_obs[key] = torch.as_tensor(self.next_observations[key][indices], device=self.device)
        
        return {
            "obs": obs,
            "next_obs": next_obs,
            "actions": torch.as_tensor(self.actions[indices], device=self.device),
            "rewards": torch.as_tensor(self.rewards[indices], device=self.device),
            "dones": torch.as_tensor(self.dones[indices], device=self.device)
        }

    def __len__(self):
        return self.buffer_size if self.full else self.pos

class OptimizedDQN:
    """
    Optimized DQN implementation with improvements to reduce Python-level loop overhead
    and more efficient caching.
    """
    def __init__(
        self,
        env,
        learning_rate=3e-4,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        target_update_interval=500,
        train_freq=1,
        gradient_steps=1,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        device="auto",
        seed=None,
        env_pool_size=1,
        optimize_memory_usage=False,
        double_q=False,
        learning_starts=100
    ):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.max_grad_norm = max_grad_norm
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Store environment
        self.env = env
        self.optimize_memory_usage = optimize_memory_usage
        
        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            self.env.reset(seed=seed)
        
        # Create Q-network
        self.q_network = CNNQNetwork().to(self.device)
        self.target_q_network = CNNQNetwork().to(self.device)
        
        # Update target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self._setup_action_mapping()
        
        # Create replay buffer
        self.replay_buffer = OptimizedReplayBuffer(
            buffer_size=buffer_size,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=self.device
        )
        print("Using Standard Experience Replay")
        
        # Environment pool for parallel collection
        self.env_pool_size = env_pool_size
        
        # Training info
        self.num_timesteps = 0
        self.num_episodes = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.updates = 0
        self.loss_history = []
        self.reward_history = []
        self.success_rate_history = []
        
        # Exploration parameters
        self.exploration_schedule = LinearSchedule(
            schedule_timesteps=int(exploration_fraction * 5000),
            initial_p=exploration_initial_eps,
            final_p=exploration_final_eps
        )
        
        # Additional features
        self.double_q = double_q
        
        # Add learning_starts attribute
        self.learning_starts = learning_starts
        
        # Add episode tracking
        self.episode_timesteps = 0
        self.last_episode_reward = 0
        self.best_return = float('-inf')
        
        # Add caching attributes
        self._last_state = None
        self._cached_valid_actions = None
        self._cached_valid_action_tensors = None
        self._q_value_cache = {}
        self._action_tensor_cache = {}
        
        # Initialize environment pool
        self.env_pool = [env]
        self.env_pool_index = 0

    def _setup_action_mapping(self):
        self._action_to_idx = {}
        self._idx_to_action = {}
        for idx, action in self.env._action_map.items():
            key = (tuple(action.pos_from), tuple(action.pos_to))
            self._action_to_idx[key] = idx
            self._idx_to_action[idx] = action
    
    def _get_action_idx(self, action):
        key = (tuple(action.pos_from), tuple(action.pos_to))
        return self._action_to_idx.get(key)
    
    def _ensure_5d_tensor(self, tensor):
        if tensor.dim() < 4:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() < 5:
            tensor = tensor.unsqueeze(1)
        return tensor
    
    def _preprocess_observation(self, obs):
        if isinstance(obs["state_zone0"], np.ndarray):
            state_zone0 = torch.FloatTensor(obs["state_zone0"]).to(self.device)
            state_zone1 = torch.FloatTensor(obs["state_zone1"]).to(self.device)
        else:
            state_zone0 = obs["state_zone0"].to(self.device)
            state_zone1 = obs["state_zone1"].to(self.device)
        state_zone0 = self._ensure_5d_tensor(state_zone0)
        state_zone1 = self._ensure_5d_tensor(state_zone1)
        return state_zone0, state_zone1
    
    def _get_action_tensor(self, action, env=None):
        key = (tuple(action.pos_from), tuple(action.pos_to))
        if key in self._action_tensor_cache:
            return self._action_tensor_cache[key]
        if env is None:
            env = self._get_env_from_pool()
        action_3d = env.action_3d(action)
        action_zone0 = torch.FloatTensor(action_3d[0]).to(self.device)
        action_zone1 = torch.FloatTensor(action_3d[1]).to(self.device)
        action_zone0 = self._ensure_5d_tensor(action_zone0)
        action_zone1 = self._ensure_5d_tensor(action_zone1)
        # Only check cache size periodically instead of on every insertion
        if len(self._action_tensor_cache) > 500:  # Increased from 100 to 500
            # Clear half the cache instead of just one entry
            keys_to_remove = list(self._action_tensor_cache.keys())[:len(self._action_tensor_cache)//2]
            for key_to_remove in keys_to_remove:
                del self._action_tensor_cache[key_to_remove]
        self._action_tensor_cache[key] = (action_zone0, action_zone1)
        return action_zone0, action_zone1
    
    def _get_q_values_for_valid_actions(self, state_zone0, state_zone1):
        """
        Vectorized computation of Q-values for all valid actions.
        Uses caching to avoid redundant computation.
        """
        # Get valid actions from the environment
        valid_actions = self.env.actions()
        if not valid_actions:
            return [], []
        
        state_key = (state_zone0.sum().item(), state_zone1.sum().item())
        if self._last_state == state_key and self._cached_valid_actions is not None:
            # Check if cached actions are still valid in the current environment state
            if all(any(ca == va for va in valid_actions) for ca in [self._idx_to_action.get(i) for i in self._cached_valid_actions]):
                return self._cached_valid_action_tensors, self._cached_valid_actions
        
        state_zone0 = self._ensure_5d_tensor(state_zone0)
        state_zone1 = self._ensure_5d_tensor(state_zone1)
        
        # Build cache keys for all valid actions.
        cache_keys = [(state_key, (tuple(action.pos_from), tuple(action.pos_to))) for action in valid_actions]
        
        # Separate cached and uncached actions.
        cached_indices = []
        not_cached_indices = []
        for i, key in enumerate(cache_keys):
            if key in self._q_value_cache:
                cached_indices.append(i)
            else:
                not_cached_indices.append(i)
        
        q_values = [None] * len(valid_actions)
        action_indices = [None] * len(valid_actions)
        
        # For cached ones, retrieve Q-values.
        for i in cached_indices:
            q_values[i] = self._q_value_cache[cache_keys[i]]
            action_idx = self._get_action_idx(valid_actions[i])
            if action_idx is None:
                for idx, act in self.env._action_map.items():
                    if act == valid_actions[i]:
                        action_idx = idx
                        break
            action_indices[i] = action_idx
        
        # For uncached ones, compute in a vectorized batch.
        if not_cached_indices:
            action_zone0_list = []
            action_zone1_list = []
            for i in not_cached_indices:
                az0, az1 = self._get_action_tensor(valid_actions[i])
                action_zone0_list.append(az0)
                action_zone1_list.append(az1)
            
            if action_zone0_list:  # Ensure we have actions to process
                batch_action_zone0 = torch.cat(action_zone0_list, dim=0)
                batch_action_zone1 = torch.cat(action_zone1_list, dim=0)
                batch_size_not_cached = len(not_cached_indices)
                tiled_state_zone0 = state_zone0.expand(batch_size_not_cached, *state_zone0.shape[1:])
                tiled_state_zone1 = state_zone1.expand(batch_size_not_cached, *state_zone1.shape[1:])
                with torch.no_grad():
                    batch_q_values = self.q_network(tiled_state_zone0, tiled_state_zone1, batch_action_zone0, batch_action_zone1)
                batch_q_values = batch_q_values.squeeze(1).tolist()
                for idx, i in enumerate(not_cached_indices):
                    q_values[i] = batch_q_values[idx]
                    # Only check cache size periodically instead of on every insertion
                    if len(self._q_value_cache) > 5000:  # Increased from 1000 to 5000
                        # Clear half the cache instead of just one entry
                        keys_to_remove = list(self._q_value_cache.keys())[:len(self._q_value_cache)//2]
                        for key_to_remove in keys_to_remove:
                            del self._q_value_cache[key_to_remove]
                    self._q_value_cache[cache_keys[i]] = batch_q_values[idx]
                    action_idx = self._get_action_idx(valid_actions[i])
                    if action_idx is None:
                        for idx2, act in self.env._action_map.items():
                            if act == valid_actions[i]:
                                action_idx = idx2
                                break
                    action_indices[i] = action_idx
        
        self._last_state = state_key
        self._cached_valid_actions = action_indices
        self._cached_valid_action_tensors = q_values
        return q_values, action_indices
    
    def _select_action(self, obs, deterministic=False):
        state_zone0, state_zone1 = self._preprocess_observation(obs)
        if deterministic or random.random() > self.exploration_rate:
            # Get valid actions from the environment first
            valid_actions = self.env.actions()
            if not valid_actions:
                return random.randrange(self.env.action_space.n), {"q_value": 0.0}
            
            # Get Q-values only for valid actions
            q_values, action_indices = self._get_q_values_for_valid_actions(state_zone0, state_zone1)
            
            # Sanity check that we have valid actions with q-values
            if not action_indices or len(action_indices) == 0:
                # If no valid actions with q-values, choose randomly from valid actions
                random_action = random.choice(valid_actions)
                action_idx = self._get_action_idx(random_action)
                if action_idx is None:
                    for idx, act in self.env._action_map.items():
                        if act == random_action:
                            action_idx = idx
                            break
                return action_idx, {"q_value": 0.0}
            
            # Choose the action with the highest Q-value
            best_idx = np.argmax(q_values)
            best_q_value = q_values[best_idx]
            best_action = action_indices[best_idx]
            
            # Verify that the selected action is valid
            chosen_action = self._idx_to_action.get(best_action)
            if chosen_action is None:
                chosen_action = self.env._action_map[best_action]
            
            if not any(chosen_action == va for va in valid_actions):
                # If somehow we've selected an invalid action, choose randomly from valid actions
                random_action = random.choice(valid_actions)
                action_idx = self._get_action_idx(random_action)
                if action_idx is None:
                    for idx, act in self.env._action_map.items():
                        if act == random_action:
                            action_idx = idx
                            break
                return action_idx, {"q_value": 0.0}
            
            return best_action, {"q_value": best_q_value}
        else:
            # For exploration, directly choose from valid actions
            valid_actions = self.env.actions()
            if not valid_actions:
                return random.randrange(self.env.action_space.n), {"q_value": 0.0}
            
            random_action = random.choice(valid_actions)
            action_idx = self._get_action_idx(random_action)
            if action_idx is None:
                for idx, act in self.env._action_map.items():
                    if act == random_action:
                        action_idx = idx
                        break
            if action_idx is None:
                # Emergency fallback - shouldn't happen
                action_idx = random.randrange(self.env.action_space.n)
            
            return action_idx, {"q_value": 0.0}
    
    def _soft_update_target_network(self, tau=0.005):
        with torch.no_grad():
            for target_param, local_param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def reward_shaping(self, obs, action, reward, next_obs, done):
        """
        Apply reward shaping to sparse rewards to improve learning
        """
        if not self.use_reward_shaping:
            return reward
            
        # If already receiving a reward, keep it
        if reward > 0:
            return reward
            
        # Check if the agent moved any box to zone 1
        state_zone1 = obs["state_zone1"]
        next_state_zone1 = next_obs["state_zone1"]
        
        # Count boxes in zone 1
        current_boxes = np.sum(state_zone1)
        next_boxes = np.sum(next_state_zone1)
        
        # If more boxes in zone 1, give bonus
        if next_boxes > current_boxes:
            bonus = 0.1 * (next_boxes - current_boxes)
            return reward + bonus
            
        # If the agent is near a box, give a small reward
        action_zone0 = obs["action_zone0"]
        if np.sum(action_zone0) > 0:  # There's a box at the action location
            return reward + 0.05
        
        return reward
    
    def train(self):
        if self.num_timesteps < self.learning_starts or len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample from replay buffer
        sample = self.replay_buffer.sample(self.batch_size)
        weights = torch.ones(self.batch_size, 1, device=self.device)
            
        obs = sample["obs"]
        next_obs = sample["next_obs"]
        actions = sample["actions"]
        rewards = sample["rewards"]
        dones = sample["dones"].view(-1, 1)
        
        # Prepare for computing Q values
        with torch.no_grad():
            next_state_zone0 = self._ensure_5d_tensor(next_obs["state_zone0"])
            next_state_zone1 = self._ensure_5d_tensor(next_obs["state_zone1"])
            next_q_values = torch.zeros(self.batch_size, 1, device=self.device)
            
            # Process next states in batches for better GPU utilization
            for batch_idx in range(0, self.batch_size, 16):  # Increased from 8 to 16
                batch_end = min(batch_idx + 16, self.batch_size)
                batch_range = range(batch_idx, batch_end)
                if torch.all(dones[batch_range]).item():
                    continue
                env = self._get_env_from_pool()
                for i in batch_range:
                    if dones[i].item():
                        next_q_values[i] = 0.0
                        continue
                    single_state_zone0 = next_state_zone0[i:i+1]
                    single_state_zone1 = next_state_zone1[i:i+1]
                    env.reset()
                    valid_actions = env.actions()
                    if not valid_actions:
                        next_q_values[i] = 0.0
                        continue
                    
                    # Implement Double Q-learning if enabled
                    if self.double_q:
                        # Get argmax actions using online network
                        max_q_value = float('-inf')
                        max_q_action_zone0 = None
                        max_q_action_zone1 = None
                        
                        # Process in larger batches for better GPU utilization
                        action_batches = [valid_actions[j:j+32] for j in range(0, len(valid_actions), 32)]
                        for action_batch in action_batches:
                            # Vectorize the forward pass for better efficiency
                            action_zone0_list = []
                            action_zone1_list = []
                            for action in action_batch:
                                az0, az1 = self._get_action_tensor(action, env)
                                action_zone0_list.append(az0)
                                action_zone1_list.append(az1)
                            
                            if action_zone0_list:
                                batch_action_zone0 = torch.cat(action_zone0_list, dim=0)
                                batch_action_zone1 = torch.cat(action_zone1_list, dim=0)
                                batch_state_zone0 = single_state_zone0.expand(len(action_batch), *single_state_zone0.shape[1:])
                                batch_state_zone1 = single_state_zone1.expand(len(action_batch), *single_state_zone1.shape[1:])
                                
                                # Use online network to select actions
                                batch_q_values = self.q_network(
                                    batch_state_zone0, batch_state_zone1, 
                                    batch_action_zone0, batch_action_zone1
                                ).squeeze(1).tolist()
                                
                                # Find action with highest Q-value
                                for idx, q_val in enumerate(batch_q_values):
                                    if q_val > max_q_value:
                                        max_q_value = q_val
                                        max_q_action_zone0 = batch_action_zone0[idx:idx+1]
                                        max_q_action_zone1 = batch_action_zone1[idx:idx+1]
                        
                        # Use target network to evaluate the selected action
                        if max_q_action_zone0 is not None:
                            next_q_values[i] = self.target_q_network(
                                single_state_zone0, single_state_zone1,
                                max_q_action_zone0, max_q_action_zone1
                            ).item()
                        else:
                            next_q_values[i] = 0.0
                    else:
                        # Standard DQN: Use target network for both action selection and evaluation
                        max_q_value = float('-inf')
                        # Process in larger batches for better GPU utilization
                        action_batches = [valid_actions[j:j+32] for j in range(0, len(valid_actions), 32)]
                        for action_batch in action_batches:
                            # Vectorize the forward pass for better efficiency
                            action_zone0_list = []
                            action_zone1_list = []
                            for action in action_batch:
                                az0, az1 = self._get_action_tensor(action, env)
                                action_zone0_list.append(az0)
                                action_zone1_list.append(az1)
                            
                            if action_zone0_list:
                                batch_action_zone0 = torch.cat(action_zone0_list, dim=0)
                                batch_action_zone1 = torch.cat(action_zone1_list, dim=0)
                                batch_state_zone0 = single_state_zone0.expand(len(action_batch), *single_state_zone0.shape[1:])
                                batch_state_zone1 = single_state_zone1.expand(len(action_batch), *single_state_zone1.shape[1:])
                                
                                batch_q_values = self.target_q_network(
                                    batch_state_zone0, batch_state_zone1, 
                                    batch_action_zone0, batch_action_zone1
                                ).squeeze(1).tolist()
                                
                                if batch_q_values:
                                    max_q_value = max(max_q_value, max(batch_q_values))
                        
                        next_q_values[i] = max_q_value if max_q_value != float('-inf') else 0.0
            
            # Apply reward scaling for non-terminal states to create a better learning signal
            # This helps with sparse rewards (where most steps have 0 reward)
            non_terminal_mask = 1 - dones
            scaled_rewards = rewards.reshape(-1, 1).clone()
            
            # Scale rewards: terminal rewards are kept as is, non-terminal rewards get small bonus
            # Add a small bonus for non-terminal rewards as a shaping reward
            scaled_rewards = scaled_rewards + 0.01 * non_terminal_mask
            
            # Use higher gamma for non-terminal states to encourage future rewards
            target_q_values = scaled_rewards + non_terminal_mask * self.gamma * next_q_values
            
            # Apply reward clipping to handle outliers and improve stability
            target_q_values = torch.clamp(target_q_values, -10.0, 100.0)
        
        state_zone0 = self._ensure_5d_tensor(obs["state_zone0"])
        state_zone1 = self._ensure_5d_tensor(obs["state_zone1"])
        
        total_loss = 0.0
        mini_batch_size = 16
        self.optimizer.zero_grad()
        
        # Track TD errors for prioritized replay
        td_errors = torch.zeros(self.batch_size, device=self.device)
        
        for batch_idx in range(0, self.batch_size, mini_batch_size):
            batch_end = min(batch_idx + mini_batch_size, self.batch_size)
            actual_batch_size = batch_end - batch_idx
            batch_state_zone0 = state_zone0[batch_idx:batch_end]
            batch_state_zone1 = state_zone1[batch_idx:batch_end]
            batch_actions = actions[batch_idx:batch_end]
            batch_targets = target_q_values[batch_idx:batch_end]
            batch_weights = weights[batch_idx:batch_end]
            
            # Vectorize action tensor retrieval
            action_zone0_list = []
            action_zone1_list = []
            env = self._get_env_from_pool()
            for i in range(actual_batch_size):
                action_idx = batch_actions[i].item()
                action = self._idx_to_action.get(action_idx)
                if action is None:
                    action = env._action_map[action_idx]
                az0, az1 = self._get_action_tensor(action, env)
                action_zone0_list.append(az0)
                action_zone1_list.append(az1)
            batch_action_zone0 = torch.cat(action_zone0_list, dim=0)
            batch_action_zone1 = torch.cat(action_zone1_list, dim=0)
            
            # Compute predictions in a single forward pass
            batch_predictions = self.q_network(batch_state_zone0, batch_state_zone1, batch_action_zone0, batch_action_zone1)
            
            # Calculate TD errors for each sample in the batch
            with torch.no_grad():
                batch_td_errors = torch.abs(batch_predictions - batch_targets)
                td_errors[batch_idx:batch_end] = batch_td_errors.squeeze()
            
            # Use Huber loss which is more robust to outliers than MSE
            batch_loss = torch.nn.functional.smooth_l1_loss(batch_predictions, batch_targets, reduction='none')
            
            # Apply importance sampling weights if using prioritized replay
            weighted_loss = batch_loss * batch_weights
            batch_loss = weighted_loss.mean()
            
            batch_loss.backward()
            total_loss += batch_loss.item() * actual_batch_size
        
        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        self.optimizer.step()
        
        avg_loss = total_loss / self.batch_size
        self.loss_history.append(avg_loss)
        return avg_loss

    def learn(self, total_timesteps, eval_freq=10000, log_freq=1000):
        obs = self.env.reset()
        self.episode_timesteps = 0
        self.last_episode_reward = 0
        print(f"Starting training for {total_timesteps} timesteps...")
        start_time = time.time()
        pbar = tqdm(total=total_timesteps, desc="Training Progress")
        
        # Tracking stats
        rewards_history = []
        invalid_actions_count = 0
        valid_actions_count = 0
        recent_rewards = []
        
        # For better exploration - anneal epsilon over time
        exploration_schedule = np.linspace(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            int(self.exploration_fraction * total_timesteps)
        )
        
        while self.num_timesteps < total_timesteps:
            # Update exploration rate based on schedule
            if self.num_timesteps < len(exploration_schedule):
                self.exploration_rate = exploration_schedule[self.num_timesteps]
            else:
                self.exploration_rate = self.exploration_final_eps
            
            action, info = self._select_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Track rewards and action validity
            rewards_history.append(reward)  # Track original reward for stats
            recent_rewards.append(reward)

            done = terminated or truncated
            self.replay_buffer.add(obs, next_obs, action, reward, done, info)
            obs = next_obs
            self.num_timesteps += 1
            self.episode_timesteps += 1
            self.last_episode_reward += reward  # Track original reward for episode return

            # Increase training frequency when we have enough data
            training_freq = max(1, self.train_freq // (1 + self.num_timesteps // 1000))  
            
            if self.num_timesteps > self.learning_starts and self.num_timesteps % training_freq == 0:
                # Train multiple times per step as we get further in training
                n_train_steps = 1 + self.num_timesteps // 2000  # Gradually increase training frequency
                for _ in range(min(4, n_train_steps)):  # Cap at a reasonable number
                    self.train()
                    
                if self.num_timesteps % self.target_update_interval == 0:
                    if self.tau < 1.0:
                        self._soft_update_target_network(self.tau)
                    else:
                        self.target_q_network.load_state_dict(self.q_network.state_dict())
            
            if self.num_timesteps % log_freq == 0:
                avg_loss = np.mean(self.loss_history[-100:]) if self.loss_history else 0
                avg_recent_reward = np.mean(recent_rewards) if recent_rewards else 0
                invalid_action_rate = invalid_actions_count / max(1, (invalid_actions_count + valid_actions_count))
                
                elapsed_time = time.time() - start_time
                fps = int(self.num_timesteps / elapsed_time)
                log_msg = (
                    f"Steps: {self.num_timesteps} | "
                    f"Episodes: {self.num_episodes} | "
                    f"Avg Loss: {avg_loss:.5f} | "
                    f"Avg Reward: {avg_recent_reward:.5f} | "
                    f"Invalid Actions: {invalid_action_rate:.2%} | "
                    f"Exploration: {self.exploration_rate:.5f} | "
                    f"FPS: {fps}"
                )
                pbar.set_description(log_msg)
                
                # Reset counters periodically
                if invalid_actions_count + valid_actions_count > 1000:
                    invalid_actions_count = 0
                    valid_actions_count = 0
            
            if self.num_timesteps % eval_freq == 0:
                self.evaluate(num_episodes=5)
            
            if done:
                self.num_episodes += 1
                self.episode_rewards.append(self.last_episode_reward)
                self.episode_lengths.append(self.episode_timesteps)
                
                # Track episode returns for learning curve
                self.reward_history.append(self.last_episode_reward)
                if len(self.reward_history) > 0:
                    self.moving_avg_return = np.mean(self.reward_history[-100:])
                
                # For successful episodes, perform additional training
                if self.last_episode_reward > 0 and self.num_timesteps > self.learning_starts:
                    # More training iterations for successful episodes to reinforce good behavior
                    n_success_train = 3  # Number of extra training steps for successful episodes
                    for _ in range(n_success_train):
                        self.train()
                
                # If this is the best episode so far, save the model
                if self.last_episode_reward > self.best_return:
                    self.best_return = self.last_episode_reward
                    if self.num_timesteps > self.learning_starts:
                        model_path = os.path.join("optimized_results", "best_model.pt")
                        self.save(model_path)
                        print(f"\nNew best return: {self.best_return:.2f}, model saved to {model_path}")
                
                # Print episode summary
                print(f"\nEpisode {self.num_episodes} completed: "
                      f"Reward={self.last_episode_reward:.2f}, "
                      f"Length={self.episode_timesteps}, "
                      f"Moving Avg={self.moving_avg_return:.2f}, "
                      f"Terminal: {terminated}, Truncated: {truncated}")
                
                obs = self.env.reset()
                self.episode_timesteps = 0
                self.last_episode_reward = 0
            
            pbar.update(1)
        pbar.close()
        print("Training complete!")
        
        # Save reward history for analysis
        np.save("reward_history.npy", np.array(rewards_history))
        
        return self
    
    def evaluate(self, num_episodes=10):
        print(f"\nEvaluating agent for {num_episodes} episodes...")
        total_rewards = []
        episode_lengths = []
        eval_env = BoxMoveEnvGym(
            horizon=self.env.horizon, 
            gamma=self.env.gamma, 
            n_boxes=self.env.n_boxes
        )
        
        # Add progress bar for evaluation
        eval_progress = tqdm(range(num_episodes), desc="Evaluation Progress")
        
        for i in eval_progress:
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            
            while not done:
                # Always use deterministic action selection with additional validation
                valid_actions = eval_env.actions()
                if not valid_actions:
                    break
                    
                # Get Q-values for valid actions
                state_zone0, state_zone1 = self._preprocess_observation(obs)
                q_values, action_indices = self._get_q_values_for_valid_actions(state_zone0, state_zone1)
                
                # Ensure action indices are not empty
                if not action_indices or len(action_indices) == 0:
                    # Choose randomly from valid actions if q-values unavailable
                    random_action = random.choice(valid_actions)
                    action_idx = self._get_action_idx(random_action)
                    if action_idx is None:
                        for idx, act in eval_env._action_map.items():
                            if act == random_action:
                                action_idx = idx
                                break
                    action = action_idx
                else:
                    # Choose best action by q-value
                    best_idx = np.argmax(q_values)
                    action = action_indices[best_idx]
                    
                    # Verify the action is valid in the current state
                    chosen_action = self._idx_to_action.get(action)
                    if chosen_action is None:
                        chosen_action = eval_env._action_map[action]
                    
                    if not any(chosen_action == va for va in valid_actions):
                        # If somehow invalid, select randomly from valid actions
                        random_action = random.choice(valid_actions)
                        action_idx = self._get_action_idx(random_action)
                        if action_idx is None:
                            for idx, act in eval_env._action_map.items():
                                if act == random_action:
                                    action_idx = idx
                                    break
                        action = action_idx
                
                # Take the action
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_steps += 1
                
                # Break if we're in an invalid state somehow
                if reward == -1:
                    eval_progress.write(f"  Warning: Invalid action detected in evaluation (episode {i+1})")
                    break
            
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            
            # Update progress bar description with current episode result
            eval_progress.set_description(f"Eval: Episode {i+1}/{num_episodes}, Reward: {episode_reward:.2f}")
            
            # Use tqdm.write instead of print to avoid breaking the progress bar
            eval_progress.write(f"  Episode {i+1}/{num_episodes}: Reward = {episode_reward:.2f}, Length = {episode_steps}")
        
        mean_reward = np.mean(total_rewards)
        mean_length = np.mean(episode_lengths)
        print(f"Evaluation complete. Mean reward: {mean_reward:.2f}, Mean length: {mean_length:.2f}")
        return mean_reward, mean_length
    
    def plot_learning_curve(self, save_path=None):
        """Plot the learning curve showing reward and loss over time."""
        if not self.reward_history:
            print("No training data to plot.")
            return
        
        import matplotlib.pyplot as plt
        
        # Create a figure with two subplots (rewards and losses)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
        
        # Plot episode returns
        episodes = range(1, len(self.reward_history) + 1)
        ax1.plot(episodes, self.reward_history, 'b-', alpha=0.3, label='Episode Return')
        
        # Calculate and plot moving average
        if len(self.reward_history) >= 10:
            window_size = min(100, len(self.reward_history) // 10)
            moving_avg = [np.mean(self.reward_history[max(0, i-window_size):i+1]) 
                         for i in range(len(self.reward_history))]
            ax1.plot(episodes, moving_avg, 'r-', label=f'Moving Average ({window_size} episodes)')
        
        ax1.set_title('Training Rewards')
        ax1.set_ylabel('Episode Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot losses if available
        if self.loss_history:
            steps = range(1, len(self.loss_history) + 1)
            ax2.plot(steps, self.loss_history, 'g-', alpha=0.3, label='Loss')
            
            # Calculate and plot moving average of losses
            if len(self.loss_history) >= 10:
                window_size = min(100, len(self.loss_history) // 10)
                moving_avg_loss = [np.mean(self.loss_history[max(0, i-window_size):i+1]) 
                                for i in range(len(self.loss_history))]
                ax2.plot(steps, moving_avg_loss, 'm-', label=f'Moving Average Loss ({window_size} steps)')
            
            ax2.set_title('Training Loss')
            ax2.set_ylabel('Loss')
            ax2.set_xlabel('Training Steps')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Use logarithmic scale for losses if there's a wide range
            if max(self.loss_history) / (min(self.loss_history) + 1e-10) > 100:
                ax2.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Learning curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save(self, path):
        state_dict = {
            'q_network': self.q_network.state_dict(),
            'target_q_network': self.target_q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_stats': {
                'num_timesteps': self.num_timesteps,
                'total_episodes': self.num_episodes,
                'loss_history': self.loss_history,
                'reward_history': self.reward_history,
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'exploration_rate': self.exploration_rate
            }
        }
        torch.save(state_dict, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        if not os.path.exists(path):
            print(f"No model found at {path}")
            return
        state_dict = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(state_dict['q_network'])
        self.target_q_network.load_state_dict(state_dict['target_q_network'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        training_stats = state_dict.get('training_stats', {})
        self.num_timesteps = training_stats.get('num_timesteps', 0)
        self.num_episodes = training_stats.get('total_episodes', 0)
        self.loss_history = training_stats.get('loss_history', [])
        self.reward_history = training_stats.get('reward_history', [])
        self.episode_rewards = training_stats.get('episode_rewards', [])
        self.episode_lengths = training_stats.get('episode_lengths', [])
        self.exploration_rate = training_stats.get('exploration_rate', self.exploration_initial_eps)
        print(f"Model loaded from {path} (timesteps: {self.num_timesteps}, episodes: {self.num_episodes})")

    def _get_env_from_pool(self):
        """Get an environment from the pool."""
        env = self.env_pool[self.env_pool_index]
        self.env_pool_index = (self.env_pool_index + 1) % len(self.env_pool)
        return env
