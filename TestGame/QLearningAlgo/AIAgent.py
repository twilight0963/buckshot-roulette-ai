from Game import Vars
from . import PlayerKnownShells ,EncodeItems, AIActions
import pickle, time

def loadQTable():
    try:
        with open("SaveStates/QTable.bin", "rb") as f:
            Vars.Q = pickle.load(f)
    except FileNotFoundError:
        print("Q-table file not found. Starting fresh training.")
    except EOFError:
        print("Q-table file is empty. Starting fresh training.")


def aiTurn(steal_mode = False):
    state = getCurrentState()
    
    actions = getAvailableActions(steal_mode)

    if not actions:
        print("No valid actions. Skipping AI turn.")
        return
    
    action = selectAction(state, actions)

    print(f"[AI] State: {state}, Selected Action: {action}")

    reward, next_state = takeAction(action)

    if Vars.is_training:
        updateQTable(state, action, reward, next_state)


def getCurrentState():
    global revealed_items
    barrel_encoded = PlayerKnownShells.getShells()
    player_items_bitmask, cuffs = EncodeItems.encode_items_presence(Vars.player_items)

    return tuple([Vars.player_health, Vars.bullet_index, PlayerKnownShells.getShells(), Vars.total_blank, Vars.total_live, Vars.dealer_health, Vars.turn, player_items_bitmask, cuffs])

def getAvailableActions(steal_mode):
    if steal_mode:
        if Vars.dealer_items:
            actions = []
            unique_items = set(Vars.dealer_items)
            unique_items.discard(9) #Cannot reuse adrenaline
            unique_items.discard(4) #Cannot steal handcuffs
            for item_id in unique_items:
                actions.append(40 + item_id) #4x is steal item
            return actions
    if Vars.isPH:
        return [0] #Handcuffed, skipped turn
    actions = []

    actions.append(1) #Shoot self
    actions.append(2) #Shoot dealer

    # Add item usage actions based on player inventory
    unique_items = set(Vars.player_items)
    if Vars.isDH != 0:
        unique_items.discard(4)
    for item_id in unique_items:
        actions.append(30 + item_id) #3x is use item

    return actions

def takeAction(action):
    reward = -0.01
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
            reward += 1
    elif action//10 == 3 or action//10 == 4:
        old_health = Vars.player_health
        old_unknown = PlayerKnownShells.getShells().count(0)
        AIActions.aiUseItems(action%10)
        new_unknown = PlayerKnownShells.getShells().count(0)
        if old_health < Vars.player_health and old_health!=Vars.max_health and 1 in Vars.player_items: #1 is item id of cigarette, used for healing
            reward += 0.3
        else:
            reward -= 0.3
        reward += (old_unknown - new_unknown) * 0.2

    next_state = getCurrentState()
    done = Vars.dealer_health == 0 or Vars.player_health == 0
    if done:
        if Vars.dealer_health == 0:
            reward += 10
            Vars.wins += 1
        else:
            reward -= 10
    reward /= 10
    Vars.reward += reward
    return reward, next_state


import random

def selectAction(state, actions):
    epsilon = max(0.1, 1 - (Vars.episode / 10000))
    Vars.alpha = max(0.1, 1.0 * (0.995 ** Vars.episode))
    #Epsilon-Greedy
    if random.random() < epsilon:
        #Explore
        return random.choice(actions)
    
    #Exploit
    q_values = [Vars.Q.get((state, a), 0) for a in actions]
    max_q = max(q_values)
    
    #If multiple actions have same Q-value, pick randomly among them
    best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
    return random.choice(best_actions)

def updateQTable(state, action, reward, next_state):

    alpha = Vars.alpha
    gamma = 0.95

    if state not in Vars.Q:
        Vars.Q[state] = {}
    old_value = Vars.Q[state].get(action, 0)
    future_rewards = max(Vars.Q.get(next_state, {}).values(), default=0)

    new_value = old_value + alpha * (reward + gamma * future_rewards - old_value)

    if state not in Vars.Q:
        Vars.Q[state] = {}
    Vars.Q[state][action] = new_value
    if Vars.episode%1000==0 and Vars.episode != 0:
        with open(f"SaveStates/QTable_{Vars.episode}.bin","wb") as fin:
            pickle.dump(Vars.Q, fin)
        with open(f"SaveStates/QTable.bin","wb") as fin:
            pickle.dump(Vars.Q, fin)


