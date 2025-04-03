from Game import TestGame, Vars
import pickle
import time

Vars.reward = 0
Vars.wins = 0
old_win_rate = 0
for i in range(200000):
    TestGame.runGame()
    Vars.episode = i
    if i%100 == 0 and i!= 0:
        with open("log.txt", "a") as f:
            f.write("\nReward on average: "+ str(Vars.reward/i))
            f.write(" || Win Rate = " + str(Vars.wins)+"%")
                        
        Vars.wins = 0

        
    