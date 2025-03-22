import random
import math
def calcProbability(bullet_index, known_shells, total_live, total_blank):
    if bullet_index in known_shells.keys():
        #Return true value of shell as "probability of dealing damage" (1 for live, 0 for blank)
        return known_shells[bullet_index]
    else:
        
        known_live = list(known_shells.values()).count(1)
        known_blank = list(known_shells.values()).count(0)

        unknown_live = total_live - known_live
        unknown_blank = total_blank - known_blank

        #Return Probability of current bullet being an unknown live
        return (unknown_live)/(unknown_live + unknown_blank)
    
def shootOther(turn, dealer_health, player_health, shells, bullet_index):
    global known_shells
    #Return in the form dealerHP, playerHP, bullet_index = shootOther()
    #Turn 0 = Player
    #Turn 1 = Dealer
    known_shells[bullet_index] = shells[bullet_index]
    if turn == 0:
        print(f"You shot {shells[bullet_index]}")
        if shells[bullet_index] == 1:
            #If shell is live, Player shoot Dealer, turn switched
            return 1, dealer_health-1, player_health, bullet_index+1
        else:
            #If shell is blank, Nothing changes and turn is switched
            return 1, dealer_health, player_health, bullet_index+1
    if turn == 1:
        print(f"Dealer shot {shells[bullet_index]}")
        if shells[bullet_index] == 1:
            #If shell is live, Dealer shoot Player, turn switched
            return 0, dealer_health, player_health-1, bullet_index+1
        else:
            #If shell is blank, Nothing changes and turn is switched
            return 0, dealer_health, player_health, bullet_index+1

def shootSelf(turn, dealer_health, player_health, shells, bullet_index):
    #Return in the form dealerHP, playerHP, bullet_index = shootSelf()
    #Turn 0 = Player
    #Turn 1 = Dealer
    if turn == 1:
        print(f"Dealer shot {shells[bullet_index]} (self)")
        if shells[bullet_index] == 1:
            #Dealer commit die
            turn = 0
            return turn, dealer_health-1, player_health, bullet_index+1
        else:
            #sacrifice denied, get another turn
            return turn, dealer_health, player_health, bullet_index+1
    if turn == 0:
        print(f"You shot {shells[bullet_index]} (self)")
        if shells[bullet_index] == 1:
            #Player commit die
            turn = 1
            return turn, dealer_health, player_health-1, bullet_index+1
        else:
            #sacrifice denied, get another turn
            return turn, dealer_health, player_health, bullet_index+1
        
def cigarette(health):
    #Heal by 1
    return health+1

def magnifyingGlass(known_shells, shells, bullet_index):
    #Reveal current shell
    known_shells[bullet_index] = shells[bullet_index] 
    return known_shells

def burnerPhone(known_shells, shells, bullet_index):
    #Reveal random shell
    reveal_shell = random.randint(bullet_index, len(shells)-1)
    known_shells[reveal_shell] = shells[reveal_shell]
    return known_shells

def handcuff(isHandcuffed):
    if not isHandcuffed:
        return True





def dealerTurn():
    global max_health, dealer_health, player_health, total_live, total_blank, known_shells, bullet_index, shells, turn, dealer_items, isDH
    if isDH:
        turn = 0
        isDH = False
    else:
        desperation = 1 - (dealer_health / max_health)
        shot_probability = calcProbability(bullet_index, known_shells, total_live, total_blank)
        next_probability = calcProbability(bullet_index+1, known_shells, total_live, total_blank)

        variation = random.randint(-2,2)
        aggression_score = shot_probability + desperation + (variation/10) * 0.3

        if dealer_health < max_health and 1 in dealer_items:
            dealer_health = cigarette(dealer_health)

        if aggression_score >= 0.5:
            turn, dealer_health, player_health, bullet_index = shootOther(turn, dealer_health, player_health, shells, bullet_index)
        else:
            turn, dealer_health, player_health, bullet_index = shootSelf(turn, dealer_health, player_health, shells, bullet_index)

def playerTurn():
    global dealer_health, player_health, bullet_index, shells, turn, player_items, isPH
    if isPH:
        turn = 1
        isPH = False
    else:
        ch = int(input("choose to shoot, 0 is self, 1 is dealer, 2 to use items\n"))
        if ch == 1:
            turn, dealer_health, player_health, bullet_index = shootOther(turn, dealer_health, player_health, shells, bullet_index)
        elif ch == 0:
            turn, dealer_health, player_health, bullet_index = shootSelf(turn, dealer_health, player_health, shells, bullet_index)
        else:
            item = int(input(str(player_items)+"\npick index \n"))
            player_items.pop(item)


max_health = random.randint(2,5)
player_items = []
dealer_items = []
player_health,dealer_health = max_health,max_health
while (dealer_health>0) and (player_health>0):
    player_items.extend([random.randint(1,7) for _ in range(4)])
    dealer_items.extend([random.randint(1,7) for _ in range(4)])
    shells = [random.randint(0, 1) for _ in range(random.randint(2,8))]
    while shells.count(1) == 0:
        shells = [random.randint(0, 1) for _ in range(random.randint(2,8))]
    total_live = shells.count(1)
    total_blank = shells.count(0)
    known_shells = {}
    print(f"{total_live} live shells. {total_blank} blank shells.")
    bullet_index = 0
    turn = 0
    while bullet_index < len(shells) and (dealer_health>0) and (player_health>0):
        if turn == 0:
            print(f"""
                --------------------------------
                bullet index: {bullet_index}
                dealer_health: {dealer_health}
                player_health: {player_health}
                --------------------------------
                  """)
            playerTurn()
        else:
            dealerTurn()

if player_health == 0:
    print("Lost!")
else:
    print("Player wins")




