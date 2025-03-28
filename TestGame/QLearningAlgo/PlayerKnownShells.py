def defineBarrel(shells_length):
    global player_known_shells
    EmptyKnown()
    for i in range(shells_length):
        player_known_shells[i] = "01"

def addKnown(index, value):
    global player_known_shells
    if value == 0:
        player_known_shells[index] = "10"
    else:
        player_known_shells[index] = "11"

def EmptyKnown():
    global player_known_shells
    player_known_shells = ['00' for _ in range(8)]

def getShells():
    return player_known_shells


