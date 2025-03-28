from Game import TestGame, Vars
from QLearningAlgo import AIAgent

AIAgent.loadQTable()
for i in range(10000):
    Vars.episode = i
    if i%100 == 0:
        if Vars.episode!= 0:
            print("Reward on average", Vars.reward/Vars.episode)
        print(i)
    TestGame.runGame()