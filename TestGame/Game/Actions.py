import random
from . import Vars
def cigarette(health):
    #Heal by 1
    if health > Vars.max_health:
        return health+1
    else:
        return health

def magnifyingGlass(shells, bullet_index):
    #Reveal current shell
    return shells[bullet_index] 

def burnerPhone(shells, bullet_index):
    #Reveal random shell
    reveal_shell = random.randint(bullet_index, len(shells)-1)
    return reveal_shell,shells[reveal_shell]

def handcuff(isHandcuffed):
    #Skip handcuffed player turn once
    if not isHandcuffed:
        return 2
    else:
        return isHandcuffed

def saw():
    #Double damage
    Vars.dmg = 2

def adrenaline(other_items, item_index):
    return other_items[item_index]
    
def inverter(bullet_index, shot_probability = 0):
    #Converts live round to blank, blank to live
    Vars.shells[bullet_index] = 1 - Vars.shells[bullet_index]
    return 1-shot_probability

def beer(bullet_index):
    #Eject current round
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
    Vars.player_known_shells[bullet_index] = shells[bullet_index]
    if turn == 0:
        if shells[bullet_index] == 1:
            #If shell is live, Player shoot Dealer, turn switched
            return 1, dealer_health-Vars.dmg, player_health, bullet_index+1
        else:
            #If shell is blank, Nothing changes and turn is switched
            return 1, dealer_health, player_health, bullet_index+1
    if turn == 1:
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
    if turn == 1:
        if shells[bullet_index] == 1:
            #Dealer commit die
            turn = 0
            return turn, dealer_health-Vars.dmg, player_health, bullet_index+1
        else:
            #sacrifice denied, get another turn
            return turn, dealer_health, player_health, bullet_index+1
    if turn == 0:
        if shells[bullet_index] == 1:
            #Player commit die
            turn = 1
            return turn, dealer_health, player_health-Vars.dmg, bullet_index+1
        else:
            #sacrifice denied, get another turn
            return turn, dealer_health, player_health, bullet_index+1