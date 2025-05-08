import random
from collections import deque
import pickle
from Game import Vars
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.old_win_rate = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        self.priorities = np.zeros(capacity)
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling weight
        self.beta_increment = 0.001
        self.epsilon = 1e-6

    def add(self, state, action, reward, next_state, done):
        if isinstance(state, torch.Tensor):
            state = state.to('cpu')
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.to('cpu')
        self.buffer.append((state, action, reward, next_state, done))
        max_priority = np.max(self.priorities) if self.size() > 0 else 1.0
        self.priorities[len(self.buffer) - 1] = max_priority

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and move to GPU
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        
        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)
    
    def save(self):
        # Save every 1000 episodes instead of 100
        if Vars.episode % 1000 == 0:
            filename = f"TestGame/DQNAlgorithm/SaveData/Traindata_{Vars.episode}.bin"
            with open(filename, "wb") as f:
                pickle.dump(self.buffer, f)
        if Vars.wins > self.old_win_rate:
            self.old_win_rate = Vars.wins
            try:
                with open ("TestGame/DQNAlgorithm/SaveData/Traindata.bin","wb") as x:
                    pickle.dump(self.buffer,x)
            except EOFError:
                pass
        print(f"Replay buffer saved at episode {Vars.episode}")  # Added logging

    def load(self):
        filename="Traindata.bin"
        try:
            with open(filename, "rb") as f:
                self.buffer = pickle.load(f)
            print(f"Replay buffer loaded from {filename}")
        except FileNotFoundError:
            print("No saved replay buffer found, starting fresh.")
