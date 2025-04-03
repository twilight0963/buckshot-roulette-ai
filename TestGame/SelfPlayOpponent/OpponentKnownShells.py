from Game import Vars

def defineBarrel(shells_length):
    EmptyKnown()
    for i in range(shells_length):
        Vars.known_shells[i] = -1

def addKnown(index, value):
        Vars.known_shells[index] = value

def EmptyKnown():
    Vars.known_shells = [-2 for _ in range(8)]

def getShells():
    return tuple(Vars.known_shells)
        


