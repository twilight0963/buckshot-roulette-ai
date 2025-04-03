from Game import Vars
import torch

def defineBarrel(shells_length):
    EmptyKnown()
    for i in range(shells_length):
        Vars.player_known_shells[i] = -1

def addKnown(index, value):
        Vars.player_known_shells[index] = value

def EmptyKnown():
    Vars.player_known_shells = [-2 for _ in range(8)]

def getShells():
    return tuple(Vars.player_known_shells)
        


