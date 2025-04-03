from Game import Vars
def encodeItems():
    items = [0 for _ in range(9)]
    for i in Vars.player_items:
        items[i] = Vars.player_items.count(i)
    return items