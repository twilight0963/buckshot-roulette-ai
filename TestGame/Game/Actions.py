import random
from . import Vars
def cigarette(health):
    #Heal by 1
    print("Cigarette used")
    if health > Vars.max_health:
        return health+1
    else:
        return health

def magnifyingGlass(shells, bullet_index):
    #Reveal current shell
    print("Magnifying glass used")
    return shells[bullet_index] 

def burnerPhone(shells, bullet_index):
    #Reveal random shell
    print("Burner phone used")
    reveal_shell = random.randint(bullet_index, len(shells)-1)
    return reveal_shell,shells[reveal_shell]

def handcuff(isHandcuffed):
    #Skip handcuffed player turn once
    print("Handcuff used")
    if not isHandcuffed:
        return 2

def saw():
    #Double damage
    print("Saw used")
    Vars.dmg = 2

def adrenaline(other_items, item_index):
    print("Adrenaline used")
    return other_items[item_index]
    
def inverter(bullet_index, shot_probability = 0):
    #Converts live round to blank, blank to live
    print("Inverter used")
    Vars.shells[bullet_index] = 1 - Vars.shells[bullet_index]
    return 1-shot_probability

def beer(bullet_index):
    #Eject current round
    print("Beer used")
    return bullet_index + 1

def expiredMeds(health):
    if round(random.random()):
        return health+2
    else:
        return health-1
    
def shootOther(turn, dealer_health, player_health, shells, bullet_index):
    #Return in the form dealerHP, playerHP, bullet_index = shootOther()
    #Turn 0 = Player
    #Turn 1 = Dealer
    Vars.known_shells[bullet_index] = shells[bullet_index]
    Vars.player_known_shells[bullet_index] = Vars.shells[Vars.bullet_index]
    if turn == 0:
        print(f"You shot {shells[bullet_index]}")
        if shells[bullet_index] == 1:
            #If shell is live, Player shoot Dealer, turn switched
            return 1, dealer_health-Vars.dmg, player_health, bullet_index+1
        else:
            #If shell is blank, Nothing changes and turn is switched
            return 1, dealer_health, player_health, bullet_index+1
    if turn == 1:
        print(f"Dealer shot {shells[bullet_index]}")
        if shells[bullet_index] == 1:
            #If shell is live, Dealer shoot Player, turn switched
            return 0, dealer_health, player_health-Vars.dmg, bullet_index+1
        else:
            #If shell is blank, Nothing changes and turn is switched
            return 0, dealer_health, player_health, bullet_index+1

def shootSelf(turn, dealer_health, player_health, shells, bullet_index):
    #Return in the form dealerHP, playerHP, bullet_index = shootSelf()
    #Turn 0 = Player
    #Turn 1 = Dealer
    Vars.known_shells[bullet_index] = shells[bullet_index]
    Vars.player_known_shells[Vars.bullet_index] = Vars.shells[Vars.bullet_index]
    with open("Bad_Moves.txt", "a") as f:
        f.write("\nBad Move")
        f.write(" || Shell Index =  " + str(Vars.bullet_index))
        f.write(" || Value =  " + str(Vars.player_known_shells))
    if turn == 1:
        print(f"Dealer shot {shells[bullet_index]} (self)")
        if shells[bullet_index] == 1:
            #Dealer commit die
            turn = 0
            return turn, dealer_health-Vars.dmg, player_health, bullet_index+1
        else:
            #sacrifice denied, get another turn
            return turn, dealer_health, player_health, bullet_index+1
    if turn == 0:
        print(f"You shot {shells[bullet_index]} (self)")
        if shells[bullet_index] == 1:
            #Player commit die
            turn = 1
            return turn, dealer_health, player_health-Vars.dmg, bullet_index+1
        else:
            #sacrifice denied, get another turn
            return turn, dealer_health, player_health, bullet_index+1