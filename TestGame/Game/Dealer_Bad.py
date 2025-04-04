import random
from . import Actions, Vars
def dealerTurn():
    if random.randint(0,1):
        Vars.turn, Vars.dealer_health, Vars.player_health, Vars.bullet_index = Actions.shootSelf(Vars.turn, Vars.dealer_health, Vars.player_health, Vars.shells, Vars.bullet_index)
    
    else:
        Vars.turn, Vars.dealer_health, Vars.player_health, Vars.bullet_index = Actions.shootOther(Vars.turn, Vars.dealer_health, Vars.player_health, Vars.shells, Vars.bullet_index)
    