from Game import TestGame, Vars
from DQNAlgorithm import AIPawn
from datetime import datetime

# Initialize the game
Vars.reward = 0
Vars.wins = 0
old_win_rate = 0

AIPawn.makeDQN()



# Main loop
for i in range(Vars.EPISODE_START, Vars.EPISODE_END+1):
    TestGame.runGame()
    Vars.episode = i
    #Every LOG_INTERVAL episodes, log the results
    if i%Vars.LOG_INTERVAL == 0 and i!= 0:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        avg_reward = Vars.reward / Vars.LOG_INTERVAL
        win_rate = Vars.wins * 100 / Vars.LOG_INTERVAL
        epsilon = Vars.epsilon
        last_loss = Vars.last_loss
        #Log the results
        with open("log.txt", "a") as f:
            f.write(f"\nEpisode: {i} || Reward on average: {avg_reward} || Win Rate = {win_rate}% || Epsilon: {epsilon} || Loss: {last_loss} || Time: {timestamp}")
                        
        #Reset the variables
        Vars.wins = 0
        Vars.reward = 0


