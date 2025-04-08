import torch
import torch.nn as nn
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
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

        if Vars.episode == 0:
            print("Starting new training session")
            self.reset_network()

        if not is_target:
            self.optimizer = optim.Adam(self.parameters(), lr=0.0003)
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99999)
            self.criterion = nn.MSELoss()
            self.replay_buffer = ReplayBuffer(buffer_capacity)
            self.batch_size = 128
            self.gamma = 0.99
            self.steal_mode = False
            self.target_network = DQN(input_size, output_size, buffer_capacity,True)
            self.update_target_network()

            #Target Network
            

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
            # Return zero tensor with appropriate shape for initial state
            return torch.zeros(21)  # Assuming output_size is 21 for available actions
        #Input layer
        state = torch.relu(self.fc1(state))
        #Hidden Layer
        state = torch.relu(self.fc2(state))
        #Output layer
        return self.fc3(state) 
    
    def chooseAction(self, state):
        actions = self.getAvailableActions()
        epsilon = max(0.1, 1.0 - Vars.episode / 50_000)
        if numpy.random.random() < epsilon:
            #Explore
            avalaible_actions = [i for i, available in enumerate(actions) if available]
            return numpy.random.choice(avalaible_actions)
        
        else:
            
            #Exploit
            q_values = self.forward(state)  # Get Q-values
            q_values = q_values.detach().numpy()  # Convert tensor to NumPy for masking
        
            q_values = [q if actions[i] else -float('inf') for i, q in enumerate(q_values)]
        
            return int(numpy.argmax(q_values))
    

    def getCurrentState(self):
        #Barrel encoded to int
        barrel_encoded = PlayerKnownShells.getShells()
        player_health_normalized = Vars.player_health / Vars.max_health
        dealer_health_normalized = Vars.dealer_health / Vars.max_health
        blanks_normalized = Vars.total_blank / len(Vars.shells)
        live_normalized = Vars.total_live / len(Vars.shells)
        bullet_index_normalized = Vars.bullet_index / len(Vars.shells)



        return torch.tensor([player_health_normalized, bullet_index_normalized, blanks_normalized, live_normalized, dealer_health_normalized, Vars.turn, *EncodeItems.encodeItems(), *barrel_encoded], dtype=torch.float32)
    
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
        if action in [1, 2]:  # Shooting actions
            total_shells = len(Vars.shells)
            if total_shells > 0:
                live_probability = Vars.total_live / total_shells
                # Penalize actions with high uncertainty (close to 0.5 probability)
                uncertainty_penalty = -0.2 * (1 - abs(live_probability - 0.5))
                reward += uncertainty_penalty
        if action == 1:
            AIActions.aiShootSelf()
            if Vars.shells[Vars.bullet_index-1] == 0:
                reward += 1
            else:
                reward -= 1
        elif action == 2:
            AIActions.aiShootOther()
            if Vars.shells[Vars.bullet_index-1] == 0:
                reward -= 1
            else:
                reward += 1
        elif action > 3:
            old_health = Vars.player_health
            old_unknown = PlayerKnownShells.getShells().count(0)
            AIActions.aiUseItems(action-13 if action>13 else action-3)
            new_unknown = PlayerKnownShells.getShells().count(0)
            if old_health < Vars.player_health and old_health!=Vars.max_health and 1 in Vars.player_items: #1 is item id of cigarette, used for healing
                reward += 0.1
            else:
                reward -= 0.1
            reward += (old_unknown - new_unknown) * 0.2

        next_state = self.getCurrentState()
        done = Vars.dealer_health == 0 or Vars.player_health == 0
        
        if done:
            if Vars.dealer_health == 0:
                reward += 5
                Vars.wins += 1
            else:
                reward -= 5

        self.replay_buffer.add(current_state,action,reward,next_state,done)
        Vars.reward += reward
        return reward, next_state
    
    def train(self, training=True):
        if not training:
            return
        # Check if the replay buffer has enough samples
        if self.replay_buffer.size() < self.batch_size:
            return
        

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.bool)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)  # Use target network
            max_next_q_values = torch.max(next_q_values, dim=1)[0]
            targets = rewards + self.gamma * max_next_q_values * (~dones)

        if Vars.episode % 100 == 0:
            self.replay_buffer.save()

        q_values = self.forward(states).gather(1, actions.unsqueeze(1)).squeeze()

        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Replace hard update with more frequent soft updates
        if Vars.episode % 50 == 0:  # More frequent, softer updates
            self.soft_update_target_network(tau=0.01)
    
        self.optimizer.step()
        self.scheduler.step()
        #Target Network Update
        if Vars.episode % 500 == 0:
            self.update_target_network()

    def reset_network(self):
        """Reset the DQN to its initial state"""
        # Reset network weights
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        
        self.apply(weight_reset)
        self.target_network.apply(weight_reset)
        
        # Reset optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.0003)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99999)
        
        # Clear replay buffer
        self.replay_buffer = ReplayBuffer(self.replay_buffer.buffer.maxlen)
        
        # Reset target network
        self.update_target_network()

