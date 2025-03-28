import random
from . import Dealer
from . import Vars
from QLearningAlgo import AIAgent

#1 - Cigarette
#2 - Magnifying Glass
#3 - Burner Phone
#4 - Handcuff
#5 - Inverter
#6 - Beer
#7 - Hand Saw
#8 - Expired Meds
#9 - Adrenaline
def runGame():
    Vars.done = False
    Vars.max_health = random.randint(2,5)
    Vars.player_items = []
    Vars.dealer_items = []
    Vars.player_health,Vars.dealer_health = Vars.max_health,Vars.max_health
    while (Vars.dealer_health>0) and (Vars.player_health>0):
        Vars.player_items.extend([random.randint(1,7) for _ in range(min(4,8-len(Vars.player_items)))])
        Vars.dealer_items.extend([random.randint(1,7) for _ in range(4)])
        Vars.shells = [random.randint(0, 1) for _ in range(random.randint(2,8))]
        while Vars.shells.count(1) == 0:
            Vars.shells = [random.randint(0, 1) for _ in range(random.randint(2,8))]
        Vars.total_live = Vars.shells.count(1)
        Vars.total_blank = Vars.shells.count(0)

        #PlayerKnownShells.defineBarrel(len(Vars.shells))
        
        print(f"{Vars.total_live} live shells. {Vars.total_blank} blank shells.")
        Vars.bullet_index = 0
        Vars.turn = 0
        while Vars.bullet_index < len(Vars.shells) and (Vars.dealer_health>0) and (Vars.player_health>0):
            Vars.dmg = 1
            if Vars.turn == 0:
                AIAgent.aiTurn()
            else:
                Dealer.dealerTurn()
    Vars.done = True
    Vars.alpha = max(0.1, 1 * (0.99 ** Vars.episode))
    Vars.episode+=1
    if Vars.player_health == 0:
        print("Lost!")
    else:
        print("Player wins")




