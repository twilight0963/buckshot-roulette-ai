from Game import Vars
def defineBarrel(shells_length):
    EmptyKnown()
    for i in range(shells_length):
        Vars.player_known_shells[i] = -1

def addKnown(index, value):
    if value == 0:
        Vars.player_known_shells[index] = 0
    else:
        Vars.player_known_shells[index] = 1

def EmptyKnown():
    Vars.player_known_shells = [-2 for _ in range(8)]

def getShells():
    return tuple(Vars.player_known_shells)

def encodeShells():
    x = Vars.shells
    y = [-1 for _ in range(8)]
    for i in range(len(x)):
        if x[i] == 0:
            y[i] = 0
        else:
            y[i] = 1
    return tuple(x)
        


