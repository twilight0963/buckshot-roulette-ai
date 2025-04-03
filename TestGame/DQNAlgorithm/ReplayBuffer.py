import random
from collections import deque
import pickle
from Game import Vars

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.old_win_rate = 0

    #Add to buffer
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    #Learn from buffer
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)
    
    def save(self):
        filename=f"TestGame/DQNAlgorithm/SaveData/Traindata_{Vars.episode}.bin"
        with open(filename, "wb") as f:
            pickle.dump(self.buffer, f)
        if Vars.wins > self.old_win_rate:
            self.old_win_rate = Vars.wins
            try:
                with open ("TestGame/DQNAlgorithm/SaveData/Traindata.bin","wb") as x:
                    pickle.dump(self.buffer,x)
            except EOFError:
                pass

    def load(self):
        filename="Traindata.bin"
        try:
            with open(filename, "rb") as f:
                self.buffer = pickle.load(f)
            print(f"Replay buffer loaded from {filename}")
        except FileNotFoundError:
            print("No saved replay buffer found, starting fresh.")
