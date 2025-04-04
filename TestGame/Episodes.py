from Game import TestGame, Vars


#Constants
EPISODE_START = 1
EPISODE_END = 200_000
LOG_INTERVAL = 100



# Initialize the game
Vars.reward = 0
Vars.wins = 0
old_win_rate = 0

# Main loop
for i in range(EPISODE_START, EPISODE_END+1):
    TestGame.runGame()
    Vars.episode = i
    #Every LOG_INTERVAL episodes, log the results
    if i%LOG_INTERVAL == 0 and i!= 0:
        #Log the results
        with open("log.txt", "a") as f:
            f.write("\nReward on average: "+ str(Vars.reward/LOG_INTERVAL))
            f.write(" || Win Rate = " + str(Vars.wins*100/LOG_INTERVAL)+"%")
                        
        #Reset the variables
        Vars.wins = 0
        Vars.reward = 0

        
    