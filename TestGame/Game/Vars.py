player_items, dealer_items, shells = [], [], []
known_shells = {}
player_health = 0
dealer_health = 0
max_health = 0
total_live = 0
total_blank = 0
bullet_index = 0
turn = 0
dmg = 0
isPH = 0  
isDH = 0  
episode = 0
alpha = 1
Q={}
reward = 0
is_training = True
wins = 0
player_known_shells = [-2 for _ in range(8)]

#Constants
EPISODE_START = 1
EPISODE_END = 200_000
LOG_INTERVAL = 100