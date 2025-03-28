def encode_items_presence(item_list):
    encoded = [0 for _ in range(8)]
    for item_id in item_list:
        if 1 <= item_id <= 7:
            encoded[item_id - 1] = 1
    return presence_to_bitmask(encoded), item_list.count(4)

def presence_to_bitmask(presence_list):
    bitmask = 0
    for i, present in enumerate(presence_list):
        if present:
            bitmask |= (1 << i)
    return bitmask