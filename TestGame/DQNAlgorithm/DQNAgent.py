import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this import
import numpy
import Game.Vars as Vars
from . import PlayerKnownShells
from . import AIActions
from . import EncodeItems
import torch.optim as optim
from DQNAlgorithm.ReplayBuffer import ReplayBuffer

class DQN(nn.Module):
    def __init__(self, input_size, output_size, buffer_capacity, is_target = False):
        super(DQN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Update input size to match getCurrentState output (24 features)
        self.fc1 = nn.Linear(24, 512).to(self.device)
        self.fc2 = nn.Linear(512, 512).to(self.device)
        self.fc3 = nn.Linear(512, output_size).to(self.device)
        self.dropout = nn.Dropout(0.2)

        if Vars.episode == 0:
            print("Starting new training session")
            self.reset_network()

        if not is_target:
            self.learning_rate = 1e-4
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
            self.max_grad_norm = 1.0
            self.target_update_freq = 1000
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, 
                patience=5000, verbose=True
            )
            self.criterion = nn.HuberLoss().to(self.device)
            self.replay_buffer = ReplayBuffer(buffer_capacity)
            self.batch_size = 256
            self.gamma = 0.99
            self.steal_mode = False
            self.target_network = DQN(input_size, output_size, buffer_capacity, True)
            self.update_target_network()
        
        self.to(self.device)

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

    def update_target_network(self):
        """Hard update target network parameters"""
        for target_param, param in zip(self.target_network.parameters(), self.parameters()):
            target_param.data.copy_(param.data)
        self.target_network.eval()

    def soft_update_target_network(self, tau=0.01):
        """Soft update target network parameters"""
        for target_param, param in zip(self.target_network.parameters(), self.parameters()):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data
            )

    def forward(self, state):
        if state is None or (isinstance(state, torch.Tensor) and state.nelement() == 0):
            return torch.zeros(21, device=self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        x = torch.relu(self.fc1(state))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        if x.shape[0] == 1:
            x = x.squeeze(0)
            
        return x
    
    def chooseAction(self, state):
        actions = self.getAvailableActions()
        # Fixed epsilon calculation and added print for debugging
        epsilon = max(0.05, 1.0 - (Vars.episode / 100_000))
        Vars.epsilon = epsilon
        if numpy.random.random() < epsilon:
            available_actions = [i for i, available in enumerate(actions) if available]
            return numpy.random.choice(available_actions)
        else:
            with torch.no_grad():
                q_values = self.forward(state)
                q_values = q_values.detach().cpu().numpy()
                q_values = [q if actions[i] else float('-inf') for i, q in enumerate(q_values)]
                return int(numpy.argmax(q_values))
    

    def getCurrentState(self):
        state = torch.zeros(24, dtype=torch.float32, device=self.device)
        
        # Normalize all values between 0 and 1
        state[0] = Vars.player_health / Vars.max_health
        state[1] = Vars.dealer_health / Vars.max_health
        state[2] = Vars.bullet_index / len(Vars.shells)
        
        # Known shells information (positions 3-12)
        known_shells = PlayerKnownShells.getShells()
        for i, shell in enumerate(known_shells[:10]):  # Limit to first 10 shells
            state[i + 3] = (shell + 2) / 3  # Normalize from [-2,1] to [0,1]
        
        # Item encoding (positions 13-22)
        item_encoding = EncodeItems.encodeItems()
        state[13:22] = torch.tensor(item_encoding, dtype=torch.float32, device=self.device)
        
        # Episode progress (position 23)
        state[23] = Vars.bullet_index / len(Vars.shells)
        
        return state
    
    def getAvailableActions(self):
        actions = [0 for _ in range(21)]
        if self.steal_mode:
            if Vars.dealer_items:
                unique_items = set(Vars.dealer_items)
                unique_items.discard(9) #Cannot steal adrenaline
                unique_items.discard(4) #Cannot steal handcuffs
                for item_id in unique_items:
                    actions[13 + item_id] = 1 #4x is steal item
                return actions
        if Vars.isPH:
            actions[0] = 1
            return actions #Handcuffed, skipped turn

        actions[1] = 1 #Shoot self
        actions[2] = 1 #Shoot dealer

        # Add item usage actions based on player inventory
        unique_items = set(Vars.player_items)
        if Vars.isDH:
            unique_items.discard(4)
        for item_id in unique_items:
            actions[3 + item_id] = 1 #3 + x is use item

        return actions

    def takeAction(self, action):
        reward = 0
        current_state = self.getCurrentState()
        current_health = Vars.player_health
        
        # Base reward for staying alive
        reward += 0.1
        
        # Reward for having higher health than opponent
        if Vars.player_health > Vars.dealer_health:
            reward += 0.2

        if action in [1, 2]:  # Shooting actions
            total_shells = len(Vars.shells)
            if total_shells > 0:
                live_probability = Vars.total_live / total_shells
                # Reward for making informed decisions
                if abs(live_probability - 0.5) > 0.2:  # More certain about outcome
                    reward += 0.3
        
        if action == 1:  # Shoot self
            AIActions.aiShootSelf()
            if Vars.shells[Vars.bullet_index-1] == 0:  # Blank
                reward += 0.5  # Reduced from 1.0
            else:  # Live
                reward -= 0.5  # Reduced from 1.0
        elif action == 2:  # Shoot dealer
            AIActions.aiShootOther()
            if Vars.shells[Vars.bullet_index-1] == 0:  # Blank
                reward -= 0.3  # Reduced penalty
            else:  # Live
                reward += 0.8
        elif action > 3:  # Item usage
            old_health = Vars.player_health
            old_unknown = PlayerKnownShells.getShells().count(0)
            AIActions.aiUseItems(action-13 if action>13 else action-3)
            new_unknown = PlayerKnownShells.getShells().count(0)
            
            # Reward for healing when low on health
            if old_health < Vars.player_health and old_health < Vars.max_health/2:
                reward += 0.4
            elif old_health < Vars.player_health:
                reward += 0.2
                
            # Reward for gathering information
            reward += (old_unknown - new_unknown) * 0.3

        next_state = self.getCurrentState()
        done = Vars.dealer_health == 0 or Vars.player_health == 0
        
        if done:
            if Vars.dealer_health == 0:
                reward += 1  
                Vars.wins += 1
            else:
                reward -= 1  

        # Penalty for losing health
        if Vars.player_health < current_health:
            reward -= 0.3

        # Normalize rewards to [-1, 1] range
        reward = numpy.clip(reward / 5.0, -1.0, 1.0)
        
        # Use reward scaling
        reward = reward * 0.1  # Scale down rewards

        self.replay_buffer.add(current_state, action, reward, next_state, done)
        Vars.reward += reward
        return reward, next_state
    
    def train(self, training=True):
        if not training or self.replay_buffer.size() < self.batch_size:
            return 0.0  # Return 0 if no training occurs
        
        total_loss = 0.0
        num_batches = min(8, self.replay_buffer.size() // self.batch_size)
        
        for _ in range(num_batches):
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.tensor(dones).float().to(self.device)
            
            # Get current Q values
            current_q_values = self(states)
            current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
            
            # Get next Q values with target network
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
            
            # Calculate loss
            loss = self.criterion(current_q_values.squeeze(), target_q_values)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()  # Add current batch loss to total
        
        avg_loss = total_loss / num_batches
        Vars.last_loss = avg_loss  # Store for logging
        return avg_loss

    def reset_network(self):
        """Reset the DQN to its initial state"""
        # Reset network weights
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        
        self.apply(weight_reset)
        
        # Only reset these if they exist (main network only)
        if hasattr(self, 'optimizer'):
            self.optimizer = optim.Adam(self.parameters(), lr=0.0003)
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99999)
            self.replay_buffer = ReplayBuffer(self.replay_buffer.buffer.maxlen)
            self.update_target_network()



