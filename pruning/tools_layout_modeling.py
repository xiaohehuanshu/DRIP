# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 10:20:29 2021

@author: SHU
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely import geometry
import shapely.ops

######搭建房间
def build_room(cord_x, cord_y, len_x, len_y, quadrant=1):
    if len_x < 0:
        len_x = 0
    else:
        pass
    if len_y < 0:
        len_y = 0
    else:
        pass
    # 第一象限原点在左下角，第二象限原点在右下角，以此类推
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


# 将包含户型信息的数值参数转换为geopandas
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
    ''' 基于geopandas构建邻接矩阵 '''
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
    ''' 基于geopandas构建邻接矩阵 '''
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


######填充面积及超界面积计算
def blank_area(room_parm, room_names_inside):
    room_storage = []
    room_storage_var = []
    boundary = np.nan

    # 遍历输入参数
    room_columns = room_parm.columns
    for i in room_columns:
        if i != 'boundary':
            info = room_parm[i].values
            # 构建房间
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
    # print(room_storage)
    # print(room_storage_var)
    room_union_all = shapely.ops.unary_union(room_storage)
    room_union_var = shapely.ops.unary_union(room_storage_var)

    inside_boundary = room_union_all.intersection(boundary[0])
    outside_boundary = room_union_var.difference(boundary[0])

    area_in = round(boundary[0].area, 0)-round(inside_boundary.area, 0)
    area_out = round(outside_boundary.area, 0)

    return area_in, area_out   #area_in 为内部未填充面积，area_out 为超出边界的面积


def shape_parm_calibration(df_info, df_parm, room_need):
    '''
    校准房间约束尺寸区间超参数
    Args:
        df_info: 房间信息
        df_parm: 房间尺寸约束超参数
        room_need: 需要生成的房间名称列表
    '''
    super_std_num = 13.5  # 最标准的房间平均面积，超参数

    names_temp = ['area_min', 'area_max', 'area_optim_min', 'area_optim_max']
    names_env = [
        'black1', 'black2', 'black3', 'black4',
        'entrance', 'entrance_sub',
        'white_south', 'white_north', 'white_west', 'white_east',
        'white_m1', 'white_m2', 'white_m3', 'white_m4',
    ]
    room_need = [i for i in room_need if (not i.startswith('white_m'))]  # 排除白体
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
    df_parm.loc[names_temp, :] = df_parm.loc[names_temp, :] * 1000000  # 房间尺寸缩放
    return df_parm

######内部矩形相交面积情况
#可相交的房间类型 kitchen, toilet, path1, path2, path3

intersect_available = ['kitchen', 'toilet_main', 'toilet2', 'path1', 'path2', 'path3']
intersect_refuse = ['black1', 'black2', 'black3', 'white_m', 'white_m2', 'white_m3', 'white_south', 'white_north', 'white_third', 'entrance']
intersect_ambiguous = ['room_main', 'room2', 'room3', 'living', 'dinner']

def intersection_area(room_parameter, adjacent_matrix):
    room_parm = room_parameter.iloc[:, 1:].copy()
    adj_matrix = adjacent_matrix.iloc[1:, 1:].copy()

    rooms_available = []   #可以相交的房间相交面积
    rooms_refuse = []      #不可以相交的房间相交面积
    rooms_ambiguous = []   #模棱两可的房间相交面积

    for i in room_parm.columns:
        if i in intersect_refuse:
            for j in adj_matrix.index:
                if (j != i + '_') and (j not in intersect_refuse):
                    area_refuse = adj_matrix[i][j][1]
                    rooms_refuse.append(area_refuse)

                    # if area_refuse !=0:
                    #     print(i, j)

                elif (j != i + '_') and (j in intersect_refuse):
                    area_refuse = adj_matrix[i][j][1]
                    rooms_refuse.append(area_refuse/2)    #除以2是因为会被算两遍

                    # if area_refuse !=0:
                    #     print(i, j)

        elif i in intersect_available:
            info = room_parm[i].values
            room = build_room(cord_x=info[0], cord_y=info[1], len_x=info[2], len_y=info[3], quadrant=1)[0]
            room_in = room.buffer(-900, cap_style=3, join_style=3)  #-900mm缓冲区
            room_out = room.difference(room_in)

            for j in room_parm.columns:
                if (j != i) and (j not in intersect_refuse) and (j not in intersect_available):
                    info1 = room_parm[j].values
                    room1 = build_room(cord_x=info1[0], cord_y=info1[1], len_x=info1[2], len_y=info1[3], quadrant=1)[0]

                    area_available_in = round(room_in.intersection(room1).area, 0)
                    rooms_available.append(area_available_in)

                    area_available_out = round(room_out.intersection(room1).area, 0)
                    rooms_available.append(area_available_out/2)    #对于可相交矩形来说，外环相交算一半面积

                    # if area_available_in != 0 or area_available_out != 0:
                    #     print(i, j)

                elif (j != i) and (j not in intersect_refuse) and (j in intersect_available):
                    info1 = room_parm[j].values
                    room1 = build_room(cord_x=info1[0], cord_y=info1[1], len_x=info1[2], len_y=info1[3], quadrant=1)[0]

                    area_available_in = round(room_in.intersection(room1).area, 0)
                    rooms_available.append(area_available_in/2)

                    area_available_out = round(room_out.intersection(room1).area, 0)
                    rooms_available.append(area_available_out/2*2)    #对于可相交矩形来说，外环相交算一半面积，除以2是因为会被算两遍

                    # if area_available_in != 0 or area_available_out != 0:
                    #     print(i, j)

                else:
                    pass

        elif i in intersect_ambiguous:
            for j in adj_matrix.index:
                if (j != i+'_') and (j in intersect_ambiguous):
                    area_ambiguous = adj_matrix[i][j][1]
                    rooms_ambiguous.append(area_ambiguous/2) #除以2是因为会被算两遍
                else:
                    pass

        else:
            print('-----')
            print(i, j)
            print('exception exist!')   #debug用的

    total_area = sum(rooms_available) + sum(rooms_refuse) + sum(rooms_ambiguous)
    # print('rooms_available: %s, rooms_refuse: %s, rooms_ambiguous: %s' %(sum(rooms_available), sum(rooms_refuse), sum(rooms_ambiguous)))

    return total_area



















