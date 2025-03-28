import random
import Player
import Dealer
import Vars

#1 - Cigarette
#2 - Magnifying Glass
#3 - Burner Phone
#4 - Handcuff
#5 - Inverter
#6 - Beer
#7 - Hand Saw
#8 - Expired Meds
#9 - Adrenaline

Vars.max_health = random.randint(2,5)
Vars.player_items = []
Vars.dealer_items = []
Vars.player_health,Vars.dealer_health = Vars.max_health,Vars.max_health
while (Vars.dealer_health>0) and (Vars.player_health>0):
    Vars.player_items.extend([random.randint(1,7) for _ in range(4)])
    Vars.dealer_items.extend([random.randint(1,7) for _ in range(4)])
    Vars.shells = [random.randint(0, 1) for _ in range(random.randint(2,8))]
    while Vars.shells.count(1) == 0:
        Vars.shells = [random.randint(0, 1) for _ in range(random.randint(2,8))]
    Vars.total_live = Vars.shells.count(1)
    Vars.total_blank = Vars.shells.count(0)
    known_shells = {}
    print(f"{Vars.total_live} live shells. {Vars.total_blank} blank shells.")
    Vars.bullet_index = 0
    Vars.turn = 0
    while Vars.bullet_index < len(Vars.shells) and (Vars.dealer_health>0) and (Vars.player_health>0):
        Vars.dmg = 1
        if Vars.turn == 0:
            print(f"""
                --------------------------------
                bullet index: {Vars.bullet_index}
                dealer_health: {Vars.dealer_health}
                player_health: {Vars.player_health}
                dealer_items: {Vars.dealer_items}
                player_items: {Vars.player_items}
                --------------------------------
                  """)
            Player.playerTurn()
        else:
            Dealer.dealerTurn()

if Vars.player_health == 0:
    print("Lost!")
else:
    print("Player wins")




