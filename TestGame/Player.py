import Vars
import Actions
def playerTurn():
    isPH = Vars.isPH
    print(isPH)
    if isPH-1 < 1:
        isPH = 0
        ch = int(input("choose to shoot, 0 is self, 1 is dealer, 2 to use items\n"))
        if ch == 1:
            Vars.turn, Vars.dealer_health, Vars.player_health, Vars.bullet_index = Actions.shootOther(Vars.turn, Vars.dealer_health, Vars.player_health, Vars.shells, Vars.bullet_index)
        elif ch == 0:
            Vars.turn, Vars.dealer_health, Vars.player_health, Vars.bullet_index = Actions.shootSelf(Vars.turn, Vars.dealer_health, Vars.player_health, Vars.shells, Vars.bullet_index)
        else:
            item = int(input(str(Vars.player_items)+"\npick index \n"))
            item = Vars.player_items[item]
            playerUseItems(item)
            Vars.player_items.remove(item)
    else:
        isPH -= 1
        Vars.turn = 1
    Vars.isPH = isPH
def playerUseItems(item, isntAdrenaline = True):
    if item == 1:
        Vars.player_health = Actions.cigarette(Vars.player_health)
    elif item == 2:
        print(Actions.magnifyingGlass(Vars.shells, Vars.bullet_index))
    elif item == 3:
        x,y = Actions.burnerPhone(Vars.shells, Vars.bullet_index)
        print(f"{x}th shell is {y}")
    elif item == 4:
        Vars.isDH = Actions.handcuff(Vars.isDH)
    elif item == 5:
        Actions.inverter(Vars.bullet_index)
    elif item == 6:
        Vars.bullet_index = Actions.beer(Vars.bullet_index)
    elif item == 7:
        Actions.saw()
    elif item == 8:
        Vars.player_health = Actions.expiredMeds(Vars.player_health)
    elif isntAdrenaline:
        a = int(input(str(Vars.dealer_items)+"\npick index \n"))
        dealer_item = Actions.adrenaline(Vars.dealer_items, dealer_item)
        playerUseItems(dealer_item, False)
        Vars.dealer_items.pop(a)
    else:
        print("Invalid index!")
        playerUseItems(item, isntAdrenaline)