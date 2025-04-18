import torch
import torch.nn as nn
import numpy
import Game.Vars as Vars
from . import OpponentKnownShells as PlayerKnownShells
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
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.criterion = nn.MSELoss()
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
        epsilon = 0.01
        if numpy.random.random() < epsilon:
            #Explore
            avalaible_actions = [i for i, available in enumerate(actions) if available]
            return numpy.random.choice(avalaible_actions)
        
        else:
            with torch.no_grad():
                #Exploit
                q_values = self.forward(state)  # Get Q-values
                q_values = q_values.detach().numpy()  # Convert tensor to NumPy for masking
            
                q_values = [q if actions[i] else -float('inf') for i, q in enumerate(q_values)]
            
                return int(numpy.argmax(q_values))
    

    def getCurrentState(self):
        #Barrel encoded to int
        barrel_encoded = PlayerKnownShells.getShells()

        return torch.tensor([Vars.dealer_health, Vars.bullet_index, Vars.total_blank, Vars.total_live, Vars.player_health, Vars.turn, *EncodeItems.encodeItems(), *barrel_encoded], dtype=torch.float32)
    
    def getAvailableActions(self):
        actions = [0 for _ in range(21)]
        if self.steal_mode:
            if Vars.player_items:
                unique_items = set(Vars.player_items)
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
        unique_items = set(Vars.dealer_items)
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
                reward -= 2
            else:
                reward += 2
        elif action > 3:
            old_health = Vars.dealer_health
            old_unknown = PlayerKnownShells.getShells().count(0)
            AIActions.aiUseItems(action-13 if action>13 else action-3)
            new_unknown = PlayerKnownShells.getShells().count(0)
            if old_health < Vars.dealer_health and old_health!=Vars.max_health and 1 in Vars.player_items: #1 is item id of cigarette, used for healing
                reward += 0.3
            else:
                reward -= 0.3
            reward += (old_unknown - new_unknown) * 0.2

        next_state = self.getCurrentState()
        done = Vars.player_health == 0 or Vars.dealer_health == 0
        
        if done:
            if Vars.player_health == 0:
                reward += 10
            else:
                reward -= 10

        self.replay_buffer.add(current_state,action,reward,next_state,done)
        Vars.reward += reward
        return reward, next_state


