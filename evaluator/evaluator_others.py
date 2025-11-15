
from shapely.ops import unary_union

def room_number_score(rooms_all, rooms_actual):
    '''
    计算是否满足房间数量要求
    '''
    rooms_need = ['white_m1', 'white_m2', 'white_m3', 'room1', 'room2', 'room3', 'room4', 'living', 'kitchen',
                  'bath1', 'bath2', 'bath1_sub', 'bath2_sub', 'dining', 'hallway', 'storeroom']
    rooms_all1 = list(set(rooms_all).intersection(set(rooms_need)))
    rooms_actual1 = list(set(rooms_all).intersection(set(rooms_actual)))
    score = len(rooms_actual1) / len(rooms_all1)
    return score

def intersection_area(df_shape):
    '''房间重叠部分面积尽量小'''
    room_shapes = df_shape.loc['rec', :].values
    rooms_rest_union = unary_union(room_shapes)
    room_union_area = rooms_rest_union.area
    overlayed_area = None
    if room_union_area != 0:
        area_tmp = []
        for poly in room_shapes:
            area_tmp.append(poly.area)
        overlayed_area = sum(area_tmp) - room_union_area
    if overlayed_area is None or room_union_area is None or room_union_area is None:
        return 0
    value = (room_union_area - 4 * overlayed_area) / room_union_area
    value = max([value, 0])
    value = min([value, 1])
    return value