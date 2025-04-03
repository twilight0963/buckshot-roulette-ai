from DQNAlgorithm.DQNAgent import DQN
import Game.Vars as Vars

dqn = DQN(23, 21, 2_000_000)

def DQNTurn():
    global dqn
    dqn.replay_buffer.load()
    if Vars.isDH != 0:
        Vars.isDH -= 1
    
    state = dqn.getCurrentState()
    action = dqn.chooseAction(state)
    print(f"[AI] State: {state}, Selected Action: {action}")
    dqn.takeAction(action)