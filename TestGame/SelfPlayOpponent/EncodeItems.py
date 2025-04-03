from Game import Vars
def encodeItems():
    items = [0 for _ in range(9)]
    for i in Vars.dealer_items:
        items[i] = Vars.dealer_items.count(i)
    return items