import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely import geometry
import shapely.ops

def build_room(cord_x, cord_y, len_x, len_y, quadrant=1):
    if len_x < 0:
        len_x = 0
    else:
        pass
    if len_y < 0:
        len_y = 0
    else:
        pass
    if quadrant == 1:
        point1 = (cord_x, cord_y)
        point2 = (cord_x+len_x, cord_y)
        point3 = (cord_x+len_x, cord_y+len_y)
        point4 = (cord_x, cord_y+len_y)

    elif quadrant == 2:
        point1 = (cord_x, cord_y)
        point2 = (cord_x, cord_y+len_y)
        point3 = (cord_x-len_x, cord_y+len_y)
        point4 = (cord_x-len_x, cord_y)

    elif quadrant == 3:
        point1 = (cord_x, cord_y)
        point2 = (cord_x-len_x, cord_y)
        point3 = (cord_x-len_x, cord_y-len_y)
        point4 = (cord_x, cord_y-len_y)

    elif quadrant == 4:
        point1 = (cord_x, cord_y)
        point2 = (cord_x, cord_y-len_y)
        point3 = (cord_x+len_x, cord_y-len_y)
        point4 = (cord_x+len_x, cord_y)

    room = geometry.Polygon([point1, point2, point3, point4])

    return room


def layout2geopandas(layout_info):
    if not isinstance(layout_info, pd.DataFrame):
        raise Exception('The input variable is not a dataframe')
    lis = []
    lis_names = []
    for i in layout_info.columns:
        room = build_room(cord_x=layout_info.loc['x', i], cord_y=layout_info.loc['y', i],
                          len_x=layout_info.loc['w', i], len_y=layout_info.loc['d', i])
        lis.append(room)
        lis_names.append(i)
    df_shape = gpd.GeoDataFrame([lis, lis], columns=lis_names, index=['rec',  'poly'])

    return df_shape


def layout_change(x, y, w, d, quadrant=1):
    if quadrant == 1:
        return x, y, w, d
    elif quadrant == 2:
        return x - w, y, w, d
    elif quadrant == 3:
        return x - w, y - d, w, d
    elif quadrant == 4:
        return x, y - d, w, d
    else:
        raise ValueError("Invalid quadrant value")


def adjacent_matrix_shapely(df_shapely):
    names = df_shapely.columns
    matrix = np.zeros((len(names), len(names)))
    df_adj = pd.DataFrame(matrix, columns=names, index=names)
    for i in df_shapely.columns:
        rectangle_i = df_shapely.loc['rec', i]
        for j in df_shapely.columns:
            if i == j:
                df_adj.loc[i, j] = -1
            else:
                rectangle_j = df_shapely.loc['rec', j]
                rectangle_i_line = rectangle_i.exterior
                rectangle_j_line = rectangle_j.exterior

                if rectangle_i.intersection(rectangle_j).area > 1:
                    df_adj.loc[i, j] = 2
                elif rectangle_i_line.intersection(rectangle_j_line).length > 1:
                    df_adj.loc[i, j] = 1
    return df_adj


def adjacent_matrix_shapely_GAT(df_shapely):
    names = df_shapely.columns
    matrix = np.zeros((len(names), len(names)))
    df_adj = pd.DataFrame(matrix, columns=names, index=names)
    for i in df_shapely.columns:
        rectangle_i = df_shapely.loc['rec', i].exterior
        for j in df_shapely.columns:
            if i == j:
                df_adj.loc[i, j] = -1
            else:
                rectangle_j = df_shapely.loc['rec', j].exterior
                if rectangle_i.intersection(rectangle_j).length > 1:
                    df_adj.loc[i, j] = 1
    return df_adj


def blank_area(room_parm, room_names_inside):
    room_storage = []
    room_storage_var = []
    boundary = np.nan

    room_columns = room_parm.columns
    for i in room_columns:
        if i != 'boundary':
            info = room_parm[i].values
            room = build_room(cord_x=info[0], cord_y=info[1], len_x=info[2], len_y=info[3], quadrant=1)
            room_storage.append(room[0])

            if i in room_names_inside:
                room_storage_var.append(room[0])
            else:
                pass

        elif i == 'boundary':
            info_b = room_parm[i].values
            boundary = build_room(cord_x=info_b[0], cord_y=info_b[1], len_x=info_b[2], len_y=info_b[3], quadrant=1)

        else:
            pass
    room_union_all = shapely.ops.unary_union(room_storage)
    room_union_var = shapely.ops.unary_union(room_storage_var)

    inside_boundary = room_union_all.intersection(boundary[0])
    outside_boundary = room_union_var.difference(boundary[0])

    area_in = round(boundary[0].area, 0)-round(inside_boundary.area, 0)
    area_out = round(outside_boundary.area, 0)

    return area_in, area_out


def shape_parm_calibration(df_info, df_parm, room_need):
    super_std_num = 13.5

    names_temp = ['area_min', 'area_max', 'area_optim_min', 'area_optim_max']
    names_env = [
        'black1', 'black2', 'black3', 'black4',
        'entrance', 'entrance_sub',
        'white_south', 'white_north', 'white_west', 'white_east',
        'white_m1', 'white_m2', 'white_m3', 'white_m4',
    ]
    room_need = [i for i in room_need if (not i.startswith('white_m'))]
    names_num = len(room_need)

    x, y, w, d = df_info.loc[:, 'boundary']
    poly_boundary = build_room(x, y, w, d)

    lis_poly_env = []
    for col in df_info.columns:
        if col in names_env:
            x1, y1, w1, d1 = df_info.loc[:, col]
            poly_env = build_room(x1, y1, w1, d1)
            lis_poly_env.append(poly_env)

    poly_env_union = shapely.ops.unary_union(lis_poly_env)
    poly_interior = poly_boundary.difference(poly_env_union)
    area_interior = poly_interior.area / 1000000
    area_average = area_interior / names_num
    area_ratio = area_average / super_std_num

    df_parm.loc[names_temp, :] = df_parm.loc[names_temp, :] * area_ratio
    df_parm.loc[names_temp, :] = df_parm.loc[names_temp, :] * 1000000
    return df_parm

















