player_items, dealer_items, shells = [], [], []
known_shells = {}
player_health, turn, dealer_health, max_health, total_live, total_blank, bullet_index, turn, dmg, isPH, isDH = 0,0,0,0,0,0,0,0,0,0,0
episode = 0
alpha = 1
Q={}
reward = 0
done=False