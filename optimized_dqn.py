import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import mse_loss
from tqdm import tqdm

# Import your environment and model components
from BoxMoveEnvGym import BoxMoveEnvGym
from QNet import CNNQNetwork

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
        env: BoxMoveEnvGym,
        learning_rate: float = 1e-4,
        buffer_size: int = 10000, 
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        device: str = "auto",
        seed: int = None,
        env_pool_size: int = 8
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.env_pool_size = env_pool_size
        self.env_pool = []
        self.env_pool_index = 0

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.env.reset(seed=seed)
        
        self.q_network = CNNQNetwork().to(self.device)
        self.target_q_network = CNNQNetwork().to(self.device)
        
        # Initialize weights with Xavier/Glorot initialization
        for module in self.q_network.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv3d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self._setup_action_mapping()
        
        self.replay_buffer = OptimizedReplayBuffer(
            buffer_size=buffer_size,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=self.device,
        )
        self.exploration_rate = exploration_initial_eps
        
        self.losses = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_episode_reward = 0
        self.episode_timesteps = 0
        self.total_episodes = 0
        self.num_timesteps = 0
        
        self._action_tensor_cache = {}
        self._initialize_env_pool()
        self._cached_valid_actions = None
        self._cached_valid_action_tensors = None
        self._last_state = None
        self._q_value_cache = {}

    def _initialize_env_pool(self):
        print(f"Creating environment pool with {self.env_pool_size} environments...")
        for _ in range(self.env_pool_size):
            new_env = BoxMoveEnvGym(
                horizon=self.env.horizon, 
                gamma=self.env.gamma, 
                n_boxes=self.env.n_boxes
            )
            new_env.reset()
            self.env_pool.append(new_env)
        print("Environment pool created.")
    
    def _get_env_from_pool(self):
        env = self.env_pool[self.env_pool_index]
        self.env_pool_index = (self.env_pool_index + 1) % self.env_pool_size
        return env
    
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
    
    def train(self):
        sample = self.replay_buffer.sample(self.batch_size)
        if self.num_timesteps < self.learning_starts or len(self.replay_buffer) < self.batch_size:
            return 0.0
        obs = sample["obs"]
        next_obs = sample["next_obs"]
        actions = sample["actions"]
        rewards = sample["rewards"]
        dones = sample["dones"].view(-1, 1)
        
        with torch.no_grad():
            next_state_zone0 = self._ensure_5d_tensor(next_obs["state_zone0"])
            next_state_zone1 = self._ensure_5d_tensor(next_obs["state_zone1"])
            next_q_values = torch.zeros(self.batch_size, 1, device=self.device)
            
            # Process in larger batches for better GPU utilization
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
                    max_q_value = float('-inf')
                    # Process in larger batches for better GPU utilization
                    action_batches = [valid_actions[j:j+32] for j in range(0, len(valid_actions), 32)]  # Increased from 16 to 32
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
            
            target_q_values = rewards.reshape(-1, 1) + (1 - dones) * self.gamma * next_q_values
        
        state_zone0 = self._ensure_5d_tensor(obs["state_zone0"])
        state_zone1 = self._ensure_5d_tensor(obs["state_zone1"])
        
        total_loss = 0.0
        mini_batch_size = 16
        self.optimizer.zero_grad()
        for batch_idx in range(0, self.batch_size, mini_batch_size):
            batch_end = min(batch_idx + mini_batch_size, self.batch_size)
            actual_batch_size = batch_end - batch_idx
            batch_state_zone0 = state_zone0[batch_idx:batch_end]
            batch_state_zone1 = state_zone1[batch_idx:batch_end]
            batch_actions = actions[batch_idx:batch_end]
            batch_targets = target_q_values[batch_idx:batch_end]
            
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
            batch_loss = mse_loss(batch_predictions, batch_targets)
            batch_loss.backward()
            total_loss += batch_loss.item() * actual_batch_size
        
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        self.optimizer.step()
        avg_loss = total_loss / self.batch_size
        self.losses.append(avg_loss)
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
        
        while self.num_timesteps < total_timesteps:
            action, info = self._select_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Track rewards and action validity
            rewards_history.append(reward)
            recent_rewards.append(reward)
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)
            
            if reward == -1:  # Invalid action penalty
                invalid_actions_count += 1
            else:
                valid_actions_count += 1
            
            done = terminated or truncated
            self.replay_buffer.add(obs, next_obs, action, reward, done, info)
            obs = next_obs
            self.num_timesteps += 1
            self.episode_timesteps += 1
            self.last_episode_reward += reward

            if self.num_timesteps > self.learning_starts and self.num_timesteps % self.train_freq == 0:
                self.train()
                if self.num_timesteps % self.target_update_interval == 0:
                    if self.tau < 1.0:
                        self._soft_update_target_network(self.tau)
                    else:
                        self.target_q_network.load_state_dict(self.q_network.state_dict())
            
            if self.num_timesteps % log_freq == 0:
                avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                avg_recent_reward = np.mean(recent_rewards) if recent_rewards else 0
                invalid_action_rate = invalid_actions_count / max(1, (invalid_actions_count + valid_actions_count))
                
                elapsed_time = time.time() - start_time
                fps = int(self.num_timesteps / elapsed_time)
                log_msg = (
                    f"Steps: {self.num_timesteps} | "
                    f"Episodes: {self.total_episodes} | "
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
                self.total_episodes += 1
                self.episode_rewards.append(self.last_episode_reward)
                self.episode_lengths.append(self.episode_timesteps)
                
                # Print episode summary
                print(f"\nEpisode {self.total_episodes} completed: "
                      f"Reward={self.last_episode_reward:.2f}, "
                      f"Length={self.episode_timesteps}, "
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
        for i in range(num_episodes):
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
                    print(f"  Warning: Invalid action detected in evaluation (episode {i+1})")
                    break
            
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            print(f"  Episode {i+1}/{num_episodes}: Reward = {episode_reward:.2f}, Length = {episode_steps}")
        
        mean_reward = np.mean(total_rewards)
        mean_length = np.mean(episode_lengths)
        print(f"Evaluation complete. Mean reward: {mean_reward:.2f}, Mean length: {mean_length:.2f}")
        return mean_reward, mean_length
    
    def plot_learning_curve(self, save_path=None):
        if not self.episode_rewards:
            print("No data to plot learning curve.")
            return
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        episodes = range(1, len(self.episode_rewards) + 1)
        ax1.plot(episodes, self.episode_rewards, 'b-')
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True, linestyle='--', alpha=0.6)
        window_size = min(10, len(self.episode_rewards))
        if window_size > 1:
            smoothed_rewards = np.convolve(self.episode_rewards, np.ones(window_size)/window_size, mode='valid')
            smoothed_episodes = range(window_size, len(self.episode_rewards) + 1)
            ax1.plot(smoothed_episodes, smoothed_rewards, 'r-', label=f'Moving Average ({window_size})', linewidth=2)
            ax1.legend()
        if self.losses:
            ax2.plot(range(1, len(self.losses) + 1), self.losses, 'g-')
            ax2.set_title('Training Loss')
            ax2.set_xlabel('Training Steps')
            ax2.set_ylabel('Loss')
            ax2.grid(True, linestyle='--', alpha=0.6)
            if max(self.losses) / (min(self.losses) + 1e-10) > 100:
                ax2.set_yscale('log')
                ax2.set_title('Training Loss (Log Scale)')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Learning curve saved to {save_path}")
        else:
            plt.show()
        plt.close(fig)
    
    def save(self, path):
        state_dict = {
            'q_network': self.q_network.state_dict(),
            'target_q_network': self.target_q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_stats': {
                'num_timesteps': self.num_timesteps,
                'total_episodes': self.total_episodes,
                'losses': self.losses,
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
        self.total_episodes = training_stats.get('total_episodes', 0)
        self.losses = training_stats.get('losses', [])
        self.episode_rewards = training_stats.get('episode_rewards', [])
        self.episode_lengths = training_stats.get('episode_lengths', [])
        self.exploration_rate = training_stats.get('exploration_rate', self.exploration_initial_eps)
        print(f"Model loaded from {path} (timesteps: {self.num_timesteps}, episodes: {self.total_episodes})")
