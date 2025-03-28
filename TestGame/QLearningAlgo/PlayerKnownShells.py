from Game import Vars
def defineBarrel(shells_length):
    EmptyKnown()
    for i in range(shells_length):
        Vars.player_known_shells[i] = "01"

def addKnown(index, value):
    if value == 0:
        Vars.player_known_shells[index] = "10"
    else:
        Vars.player_known_shells[index] = "11"

def EmptyKnown():
    Vars.player_known_shells = ['00' for _ in range(8)]

def getShells():

    return tuple(Vars.player_known_shells)



