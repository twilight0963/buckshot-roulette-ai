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
    def __init__(self, input_size, output_size, buffer_capacity):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9999)
        self.criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = 64
        self.gamma = 0.99
        self.steal_mode = False

    def forward(self, state):
        #Input layer
        state = torch.relu(self.fc1(state))
        #Hidden Layer
        state = torch.relu(self.fc2(state))
        #Output layer
        return self.fc3(state) 
    
    def chooseAction(self, state):
        actions = self.getAvailableActions()
        epsilon = 0.1 + (0.9 * numpy.exp(-0.00002 * Vars.episode))
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

        return torch.tensor([Vars.player_health, Vars.bullet_index, Vars.total_blank, Vars.total_live, Vars.dealer_health, Vars.turn, *EncodeItems.encodeItems(), *barrel_encoded], dtype=torch.float32)
    
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
        reward = -0.01
        current_state = self.getCurrentState()
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
                reward += 10
                Vars.wins += 1
            else:
                reward -= 10

        self.replay_buffer.add(current_state,action,reward,next_state,done)
        Vars.reward += reward
        return reward, next_state
    
    def train(self):
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
            next_q_values = self.forward(next_states)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]
            targets = rewards + self.gamma * max_next_q_values.detach() * (~dones)

        if Vars.episode % 100 == 0:
            self.replay_buffer.save()

        q_values = self.forward(states).gather(1, actions.unsqueeze(1)).squeeze()

        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

