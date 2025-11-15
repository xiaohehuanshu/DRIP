import geopandas as gpd
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union


# 得到公共空间几何轮廓
def get_public_space(gpd_all):
    '''
    :param gpd_all: 整个户型的全部geopandas dataframe数据，包括白体黑体等
    :return:
    '''
    rooms_all = list(gpd_all.columns)
    rooms_to_exclude = ['living', 'dining']
    rooms_left = list(set(rooms_all).difference(set(rooms_to_exclude+['boundary'])))
    shapes_to_excluded = unary_union(gpd_all.loc['rec', rooms_left])
    shape_boundary = gpd_all.loc['rec', 'boundary']
    pub = shape_boundary.difference(shapes_to_excluded)
    return pub

# 获取过道
def get_path(gpd_all):
    without_boundary = gpd_all.drop(columns=['boundary']).copy()
    boundary = gpd_all.loc['rec', 'boundary']
    rooms_as_path = list(['living', 'dining'])
    pub_rooms = list(set(gpd_all.columns.tolist()).intersection(set(rooms_as_path)))  # 筛选出存在的可做过道的房间

    without_living_dining = without_boundary.drop(columns=pub_rooms).copy()
    shapes_union = unary_union(without_boundary.loc['poly'].values)
    path_pure = boundary.difference(shapes_union)
    path_all = boundary.difference(unary_union(without_living_dining.values))

    return path_pure, path_all


# 清除多边形上的多余点（多边形上的非拐点）
def clean_polygon_midpoints(shape):
    '''
    :param shape: 闭合多段线，polygon
    :return:  已清除非拐点的多段线
    '''
    room_shape_coord = list(shape.exterior.coords)
    recorder = []
    for i in range(1, len(room_shape_coord) - 1):
        x1 = room_shape_coord[i - 1][0]
        x2 = room_shape_coord[i][0]
        x3 = room_shape_coord[i + 1][0]

        y1 = room_shape_coord[i - 1][1]
        y2 = room_shape_coord[i][1]
        y3 = room_shape_coord[i + 1][1]

        if (x1 == x2 == x3) or (y1 == y2 == y3):
            recorder.append(i)

    for j in reversed(recorder):
        del room_shape_coord[j]

    room_shape_clean = Polygon(room_shape_coord)
    return room_shape_clean