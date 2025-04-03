from Game import Vars,Actions
from . import OpponentKnownShells

def aiShootOther():
    Vars.turn, Vars.dealer_health, Vars.player_health, Vars.bullet_index = Actions.shootOther(Vars.turn, Vars.dealer_health, Vars.player_health, Vars.shells, Vars.bullet_index)
def aiShootSelf():
    Vars.turn, Vars.dealer_health, Vars.player_health, Vars.bullet_index = Actions.shootSelf(Vars.turn, Vars.dealer_health, Vars.player_health, Vars.shells, Vars.bullet_index)

def aiUseItems(item, isntAdrenaline = True):
    if item == 1:
        Vars.dealer_health = Actions.cigarette(Vars.dealer_health)
    elif item == 2:
        shell = Actions.magnifyingGlass(Vars.shells, Vars.bullet_index)
        OpponentKnownShells.addKnown(Vars.bullet_index,shell)
    elif item == 3:
        x,y = Actions.burnerPhone(Vars.shells, Vars.bullet_index)
        OpponentKnownShells.addKnown(x,y)
    elif item == 4:
        Vars.isPH = Actions.handcuff(Vars.isPH)
    elif item == 5:
        Actions.inverter(Vars.bullet_index)
    elif item == 6:
        Vars.bullet_index = Actions.beer(Vars.bullet_index)
    elif item == 7:
        Actions.saw()
    elif item == 8:
        Vars.dealer_health = Actions.expiredMeds(Vars.dealer_health)
    Vars.dealer_items.remove(item)