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
    def __init__(self, input_size, output_size, buffer_capacity, is_target = False):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        
        # Load the best model weights
        self.load_state_dict(torch.load('trainingdata.bin'))
        self.eval()  # Set to evaluation mode
        self.steal_mode = False

    # Remove training-related methods and keep only inference methods
    def forward(self, state):
        #Input layer
        state = torch.relu(self.fc1(state))
        #Hidden Layer
        state = torch.relu(self.fc2(state))
        #Output layer
        return self.fc3(state) 
    
    def chooseAction(self, state):
        actions = self.getAvailableActions()
        with torch.no_grad():
            q_values = self.forward(state)
            q_values = q_values.detach().numpy()
            q_values = [q if actions[i] else -float('inf') for i, q in enumerate(q_values)]
            return int(numpy.argmax(q_values))

    def getCurrentState(self):
        barrel_encoded = PlayerKnownShells.getShells()
        dealer_health_normalized = Vars.dealer_health / Vars.max_health 
        player_health_normalized = Vars.player_health / Vars.max_health
        blanks_normalized = Vars.total_blank / len(Vars.shells)
        live_normalized = Vars.total_live / len(Vars.shells)
        bullet_index_normalized = Vars.bullet_index / len(Vars.shells)

        return torch.tensor([dealer_health_normalized, bullet_index_normalized, blanks_normalized, live_normalized, player_health_normalized, Vars.turn, *EncodeItems.encodeItems(), *barrel_encoded], dtype=torch.float32)
    
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
        # Keep action execution but remove training-related parts
        reward = 0
        if action == 1:
            AIActions.aiShootSelf()
        elif action == 2:
            AIActions.aiShootOther()
        elif action > 3:
            AIActions.aiUseItems(action-13 if action>13 else action-3)
        
        # No need to store in replay buffer or calculate rewards
        return self.getCurrentState()


