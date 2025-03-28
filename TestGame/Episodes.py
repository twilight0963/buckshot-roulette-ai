from Game import TestGame, Vars
from QLearningAlgo import AIAgent

AIAgent.loadQTable()
for i in range(100000):
    Vars.episode = i
    if i%100 == 0:
        if Vars.episode!= 0:
            print("\n\n----++++Reward on average++++----\n\n", Vars.reward/Vars.episode)
            with open("log.txt", "a") as f:
                f.write("\nReward on average: "+ str(Vars.reward/Vars.episode))
        print(i)
    TestGame.runGame()