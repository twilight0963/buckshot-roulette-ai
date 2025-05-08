from DQNAlgorithm.DQNAgent import DQN
import Game.Vars as Vars

def makeDQN():
    global dqn
    dqn = DQN(23, 21, 2_000_000)
    dqn.replay_buffer.load()

def DQNTurn():
    if Vars.isPH != 0:
        Vars.isPH -= 1
    
    global dqn
    state = dqn.getCurrentState()
    action = dqn.chooseAction(state)

    dqn.takeAction(action)
    dqn.train()