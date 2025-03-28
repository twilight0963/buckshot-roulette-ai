import random
from . import Actions, Vars
def calcProbability(bullet_index, known_shells, total_live, total_blank):
    if bullet_index in known_shells.keys():
        #Return true value of shell as "probability of dealing damage" (1 for live, 0 for blank)
        return known_shells[bullet_index]
    else:
        
        known_live = list(known_shells.values()).count(1)
        known_blank = list(known_shells.values()).count(0)

        unknown_live = total_live - known_live
        unknown_blank = total_blank - known_blank

        #Return Probability of current bullet being an unknown live
        return (unknown_live)/(unknown_live + unknown_blank)


def shouldUseItems(next_probability, items):
    if Vars.dealer_health < Vars.max_health:
        if 1 in Vars.dealer_items:
            return 1
        elif 8 in Vars.dealer_items and Vars.dealer_health > 1:       
            return 8

    if (0.5 - shot_probability) <= 0.1 and 2 in items:
        return 2
    
    if shot_probability <= 0.5 and 5 in items:
        return 5
        
    if len(Vars.known_shells)<len(Vars.shells) and 3 in items:
        return 3

    if next_probability >= shot_probability and abs(0.5 - shot_probability) <= 0.1 and 4 in items:
        return 4
    elif next_probability > shot_probability and shot_probability <= 0.5 and 6 in items:
        return 6

    if aggression_score > 0.5 and 7 in items:
        return 7
    
    return False


def useItem(desperation, next_probability, items, isntAdrenaline = True):
    global shot_probability, aggression_score

    if 9 in Vars.dealer_items and shouldUseItems(next_probability, player_items) and isntAdrenaline:
        player_items = useItem(desperation, next_probability, player_items, False)

    if Vars.dealer_health < Vars.max_health:
        if 1 in Vars.dealer_items:
            Vars.dealer_health = Actions.cigarette(Vars.dealer_health)
            items.remove(1)
        elif 8 in Vars.dealer_items and Vars.dealer_health > 1:       
            Vars.dealer_health = Actions.expiredMeds(Vars.dealer_health)
            items.remove(8)

    if (0.5 - shot_probability) <= 0.1 and 2 in items:
        items.remove(2)
        Vars.known_shells[Vars.bullet_index] = Actions.magnifyingGlass(Vars.shells,Vars.bullet_index)
    
    if shot_probability <= 0.5 and 5 in items:
        shot_probability = Actions.inverter(Vars.bullet_index, shot_probability)
        if Vars.bullet_index in Vars.known_shells.keys():
            Vars.known_shells[Vars.bullet_index] = 1 - Vars.known_shells[Vars.bullet_index]
        variation = random.randint(-2,2)
        aggression_score = shot_probability + desperation + (variation/10) * 0.3
        items.remove(5)
        
    if len(Vars.known_shells)<len(Vars.shells) and 3 in items:
        items.remove(3)
        x,y = Actions.burnerPhone(Vars.shells,Vars.bullet_index)
        Vars.known_shells[x] = y


    if next_probability >= shot_probability and abs(0.5 - shot_probability) <= 0.1 and 4 in items:
        items.remove(4)
        Vars.isPH = Actions.handcuff(Vars.isPH)
    elif next_probability > shot_probability and shot_probability <= 0.5 and 6 in items:
        items.remove(6)
        Vars.bullet_index = Actions.beer(Vars.bullet_index)

    if aggression_score > 0.5 and 7 in items:
        items.remove(7)
        Actions.saw()
    return items


def dealerTurn():
    global shot_probability, aggression_score
    print(Vars.isDH)
    if Vars.isDH-1 < 1:
        Vars.isDH = 0
        desperation = 1 - (Vars.dealer_health / Vars.max_health)
        shot_probability = calcProbability(Vars.bullet_index, Vars.known_shells, Vars.total_live, Vars.total_blank)
        next_probability = calcProbability(Vars.bullet_index+1, Vars.known_shells, Vars.total_live, Vars.total_blank)
        variation = random.randint(-2,2)
        aggression_score = shot_probability + desperation + (variation/10) * 0.3

        Vars.dealer_items = useItem(desperation, next_probability, Vars.dealer_items)


        
        if aggression_score >= 0.5:
            Vars.turn, Vars.dealer_health, Vars.player_health, Vars.bullet_index = Actions.shootOther(Vars.turn, Vars.dealer_health, Vars.player_health, Vars.shells, Vars.bullet_index)
        else:
            Vars.turn, Vars.dealer_health, Vars.player_health, Vars.bullet_index = Actions.shootSelf(Vars.turn, Vars.dealer_health, Vars.player_health, Vars.shells, Vars.bullet_index)
    else:
        Vars.isDH -= 1
        Vars.turn = 0
