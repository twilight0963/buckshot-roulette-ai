from DQNAlgorithm.DQNAgent import DQN
import Game.Vars as Vars

dqn = DQN(23, 21, 2_000_000)

def DQNTurn():
    if Vars.isPH != 0:
        Vars.isPH -= 1
    
    global dqn
    state = dqn.getCurrentState()
    action = dqn.chooseAction(state)
    print(f"[AI] State: {state}, Selected Action: {action}")
    dqn.takeAction(action)
    dqn.train()