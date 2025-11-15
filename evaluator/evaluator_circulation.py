import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import math
import shapely
from shapely.geometry import LineString, Polygon, Point, MultiLineString
from shapely import geometry
from itertools import combinations

import networkx as nx
import itertools

import ast  # 用于安全地将字符串转换为元组或列表
from tools_layout_modeling import build_room, layout2geopandas, adjacent_matrix_shapely
from .tools_space_analysis import get_path
from .evaluator_room_shape import RoomShapeScore
from .evaluator_path_shape import PathShapeScore
pd.set_option('expand_frame_repr', False)

def normalization_circulation(func):
    def wrapper(*args, **kwargs):
        # 在函数调用前执行额外的操作
        result = func(*args, **kwargs)
        # print(result[1])
        total_score = result / 5.6
        # 在函数调用后执行额外的操作
        if isinstance(total_score, (int, float)):
            # total_score = min([total_score, 9.7])
            # total_score = max([total_score, 0])
            total_score = total_score
        else:
            total_score = 0
        return total_score
    return wrapper

@normalization_circulation
def circulation_evaluator(gpd_rooms, gpd_all, df_weight_absolute, df_weight_relative):
    '''
    :param gpd_rooms: 所有内部房间的geopandas dataframe数据
    :param gpd_all: 整个户型的全部geopandas dataframe数据，包括白体黑体等
    :return:
    '''

    # 绘图
    # layout_shapes = gpd_rooms
    # polygons_not_overlap = {}
    # for i in gpd_rooms.columns:
    #     polygons_not_overlap[i] = layout_shapes.loc['poly', i]
    # plot_shapes(polygons=list(polygons_not_overlap.values()))

    # 定义有效房间： room 多边形
    # rooms 是代表房间的多边形的合集，包括入口多边形、客厅、餐厅、厨房、卫生间、各房间；rooms_name是与rooms严格对应的房间名称列表，确保顺序一致；
    # spaces 则是同时包含房间名称和多边形顶点信息的字典，不仅包括rooms的所有房间，还包括passway过道。

    rlis = gpd_all.columns.tolist()  # 全部空间单元名称列表
    rooms_private = [
        'room1', 'room2', 'room3', 'room4', 'bath1', 'bath2',
        'storeroom', 'study_room'
    ]
    rooms_name_all = ['living', 'dining', 'kitchen', 'hallway'] + rooms_private

    room_private_exist = [i for i in rooms_name_all if i in gpd_all.columns]
    rooms_name = room_private_exist + ['entrance']
    rooms = [gpd_all.loc['poly', i] for i in rooms_name]

    # 定义所有空间
    nvs = zip(rooms_name, [list(room.exterior.coords) for room in rooms])
    # for name, value in nvs:
    #     print(name, value)
    spaces = dict((name, value) for name, value in nvs)  # 所有空间的坐标点形式
    passway, pub = get_path(gpd_all)  # 得到过道和公共空间

    # 定义过道空间
    if passway.geom_type == 'Polygon':
        iteration_path = [passway]
    else:
        iteration_new = passway.geoms
        iteration_path = [i for i in iteration_new]
    for i, poly in enumerate(iteration_path):
        spaces['passway' + str(i)] = list(poly.exterior.coords)

    # 筛选出所有过道名称-
    se = pd.Series(spaces.keys())
    se_bool = se.apply(lambda x: x.startswith('passway'))
    passway_list = se[se_bool].tolist()

    # 按照偏好顺序获取房间的邻接线（输入：所有房间多边形，目标房间，偏好顺序）,邻接线指潜在开门的边
    dic_open_door_order = {
        'kitchen': ['dining'] + passway_list + ['living', 'hallway'],
        'bath1': passway_list + ['living', 'dining', 'hallway'],
        'bath2': passway_list + ['living', 'dining', 'hallway'],
        'room1': passway_list + ['living', 'dining', 'hallway'],
        'room2': passway_list + ['living', 'dining', 'hallway'],
        'room3': passway_list + ['living', 'dining', 'hallway'],
        'room4': passway_list + ['living', 'dining', 'hallway'],
        'living': passway_list + ['entrance', 'dining', 'hallway'],
        'dining': passway_list + ['entrance', 'living', 'hallway'],
        'entrance': passway_list + ['living', 'dining', 'hallway'],
        'storeroom': passway_list + ['living', 'dining', 'hallway'],
        'hallway': passway_list + ['entrance', 'living', 'dining'],
        'study_room': passway_list + ['living', 'dining', 'hallway']
    }

    doors_dict = {}
    rooms_to_judge = list(set(rooms_name_all + ['entrance']).intersection(set(rlis)))
    for i in rooms_to_judge:
        open_door_order_exist = []  # 找到存在的过道单元
        for r in dic_open_door_order[i]:
            if (r in gpd_all.columns) or r.startswith('passway'):
                open_door_order_exist.append(r)
        # print(open_door_order_exist)
        if i in ['hallway', 'living', 'dining']:  # 如果是客厅餐厅，去几何中心点
            center_point = gpd_rooms.loc['poly', i].centroid
            doors_dict[i] = [list(center_point.coords)]
        else:
            adjacent_edges = find_adjacent_edges(outline=spaces, key_room=i, adjacent_preferences=open_door_order_exist)
            if adjacent_edges:  # 判断输出的邻接边是否是空(既该房间是否能对过道开门）
                doors_dict[i] = [list(get_midpoint(j).coords) for j in adjacent_edges]  #找到开门点
    # print(doors_dict)
    # 外轮廓的点集points
    # 定义过道空间
    if pub.geom_type == 'Polygon':
        iteration_pub = [pub]
    else:
        iteration_new = pub.geoms
        iteration_pub = [i for i in iteration_new]
    points_out = []
    for p in iteration_pub:
        points_out = points_out + list(p.exterior.coords)
    points_out = list(set(points_out))  # 先将列表转换为集合，因为集合是不重复的，故直接删除重复元素

    # 生成开门组合
    doors_dict_simple = select_proper_door_point(doors_dict, pub, points_out, gpd_all, rooms_private)  # 简化开门可能性
    combinations_doors = custom_product_dict(doors_dict_simple, prefix=None)  # 递归求解
    # combinations_doors = custom_product_dict(doors_dict, prefix=None)  # 递归求解
    # print('combinations nums:', len(combinations_doors))

    ### 计算分数：
    highest_score = 0
    best_df_paths = None
    best_door_points = []
    best_dic_absolute = {}
    best_dic_relative = {}
    super_distance = 0  # 路径距离超参数
    for i in range(len(combinations_doors)):
        room_door_points_coords = combinations_doors[i]
        door_points = [coords for _, coords in room_door_points_coords]  # 只有点

        # 创建graph的边
        graph_edges_linestring = create_graph_edges(pub, door_points, points_out)
        ####### 正式生成graph并求解 #######
        df_paths = generate_path_dataframe(room_door_points_coords, graph_edges_linestring, gpd_all)

        out_absolute = circulations_absolute_score(df_graph=df_paths, gpd_all=gpd_all, gpd_rooms=gpd_rooms, df_weight_absolute=df_weight_absolute)
        out_relative = circulations_relative_score(df_graph=df_paths, gpd_all=gpd_all, gpd_rooms=gpd_rooms, df_weight_relative=df_weight_relative)
        score_abs = out_absolute[0]
        score_rela = out_relative[0]
        score_total = score_abs + score_rela
        if score_total > highest_score:
            highest_score = score_total
            best_df_paths = df_paths
            best_door_points = door_points
            best_dic_absolute = out_absolute[1]
            best_dic_relative = out_relative[1]
            super_distance = out_absolute[2]

    # print('best_dic_absolute:', best_dic_absolute)
    # print('best_dic_relative:', best_dic_relative)
    # print(best_df_paths)
    # 绘图
    # 画图预处理 df_paths
    # bb_df_polygons, bb_df_lines, bb_df_points = preprocess_df_paths_for_plot(best_df_paths)
    # plot_shapes(polygons=[pub], lines=bb_df_lines, points=best_door_points)

    # 计算所有房间总路径长度
    # if (not best_df_paths is None) and (not gpd_rooms is None):
    #     se_length = best_df_paths.loc[:, 'Total Length']
    #     total_length = sum(se_length.values)
    #     average_length = total_length / len(se_length.index)
    #     # print(average_length)
    #     # print(super_distance)
    #     length_score = curve_downward_2parm(x=average_length, x_min=super_distance * 0.3, x_max=super_distance * 0.5) * 2
    # else:
    #     length_score = 0
    # print(length_score, '----')
    if len(gpd_rooms.columns) == 0:
        return 0
    highest_score = highest_score / len(gpd_rooms.columns)
    return highest_score

def select_proper_door_point(doors_dict, pub, points_out, gpd_all, rooms_private):
    """通过各个房间的所有可开门点位与主入口距离关系筛选出最终的开门点集合"""
    dic_doors_new = {}
    for room in doors_dict.keys():
        if not room.startswith('entrance'):
            # print(doors_dict)
            # print(room)
            # print('---------------')
            dic_sub = {key: doors_dict[key] for key in ['entrance', room]}
            combinations_doors_sub = custom_product_dict(dic_sub, prefix=None)
            if room in rooms_private:  # 私密性的房间离入口越远越好
                record = 0
            else:
                record = 10000000000
            for i in range(len(combinations_doors_sub)):
                room_door_points_coords = combinations_doors_sub[i]
                door_points = [coord for _, coord in room_door_points_coords]  # 只有点
                door_point_room_now = [coord for r, coord in room_door_points_coords if not r.startswith('entrance')]
                graph_edges_linestring = create_graph_edges(pub, door_points, points_out)
                df_paths = generate_path_dataframe(room_door_points_coords, graph_edges_linestring, gpd_all)
                if (room in rooms_private) and (not df_paths.empty):
                    length = df_paths.loc[:, 'Total Length'].values[0]
                    if length > record:
                        dic_doors_new[room] = [[door_point_room_now[0]]]
                        record = length
                elif (room not in rooms_private) and (not df_paths.empty):
                    length = df_paths.loc[:, 'Total Length'].values[0]
                    if length < record:
                        dic_doors_new[room] = [[door_point_room_now[0]]]
                        record = length
        else:
            dic_doors_new[room] = doors_dict[room]
    return dic_doors_new

def circulations_relative_score(df_graph, gpd_all, gpd_rooms, df_weight_relative):
    '''通过连续变量计算各个房间的相对流线关系得分(主要为相对主入口的关系得分）'''
    dic_score = {} # 存储每个房间得分
    if df_graph.empty:
        for col in gpd_rooms.columns:
            dic_score[col] = 0
        return 0, dic_score

    room_names_exist = gpd_rooms.columns
    room_names_exist1 = gpd_rooms.columns.tolist() + ['entrance']
    for col in df_weight_relative.columns:  # col 为 entrance
        if col in room_names_exist1:
            se_sub = df_weight_relative.loc[:, col]
            se_sub = se_sub.dropna()
            se_positive = se_sub[se_sub > 0]  # 吸引力房间
            se_negative = se_sub[se_sub < 0]  # 斥力房间
            for i in se_positive.index:  # i为吸引力房间
                for j in se_negative.index:  # j为斥力房间
                    route_positive = df_graph[(df_graph['Start Name'].str.contains(col)) & (df_graph['End Name'].str.contains(i))]
                    route_negative = df_graph[(df_graph['Start Name'].str.contains(col)) & (df_graph['End Name'].str.contains(j))]
                    if (len(route_positive) > 0) and (len(route_negative) > 0):  # 确定图中包含了这一条路径（即这两个房间之间的流线是畅通的）
                        segments_positive = route_positive['Number of Segments'].values[0]  # 两个房间之间最短路线的转折次数（经过节点数）
                        distance_positive = route_positive['Total Length'].values[0] + segments_positive * 0.5  # 如果两个房间之间出现拐角，则房间之间拓扑距离为实际距离×系数
                        segments_negative = route_negative['Number of Segments'].values[0]
                        distance_negative = route_negative['Total Length'].values[0] + segments_negative * 0.5
                        # print(i, j, distance_positive, distance_negative)
                        if distance_positive < distance_negative:  # 吸引力房间距主入口距离小于斥力房间距主入口距离
                            dic_score[str(col) + '_' + str(i) + '_' + str(j)] = 1 * (abs(df_weight_relative.loc[j, col]) + abs(df_weight_relative.loc[i, col]))
                        else:
                            dic_score[str(col) + '_' + str(i) + '_' + str(j)] = 0
    total_score = sum(dic_score.values())
    return total_score, dic_score


def circulations_absolute_score(df_graph, gpd_all, gpd_rooms, df_weight_absolute):
    '''
    通过连续变量计算各个房间的绝对流线关系得分
    :param df_graph: 流线图
    :return:
    '''
    dic_score = {} # 存储每个房间得分
    if df_graph.empty:
        for col in gpd_rooms.columns:
            dic_score[col] = 0
        return 0, dic_score

    # 通过entrance与boundary四个顶点最远点的距离，计算房间之间引力与斥力函数的超参数
    entrance = gpd_all.loc['rec', 'entrance']
    point_coords = list(gpd_all.loc['rec', 'boundary'].exterior.coords)
    points = [Point(p) for p in point_coords]
    distance = [p.distance(entrance) for p in points]
    distance_max = max(distance)
    cri_push_max = distance_max * (2/3)  # 斥力超参数
    cri_pull_max = distance_max * (2/3)  # 引力超参数
    cri_pull_min = distance_max * (1/7)  # 引力超参数
    # print(cri_length_far, cri_length_near_min)

    def calculate_score_distance(col, id, name_replace=None):
        se_screen = df_graph[(df_graph['Start Name'].str.contains(col)) & (df_graph['End Name'].str.contains(id))]
        score_distance = None
        if len(se_screen) > 0:  # 确定图中包含了这一条路径（即这两个房间之间的流线是畅通的）
            segments_num = se_screen['Number of Segments'].values[0]  # 两个房间之间最短路线的转折次数（经过节点数）
            distance = se_screen['Total Length'].values[0] + segments_num * 0.5  # 如果两个房间之间出现拐角，则房间之间拓扑距离为实际距离×系数
            if name_replace==None:
                weight = df_weight_absolute.loc[id, col]
            else:
                weight = df_weight_absolute.loc[name_replace, col]
            if weight < 0:  # 表示相斥，距离越远分越高，使用上扬曲线
                score_distance = curve_upward_2parm(x=distance, x_min=0, x_max=cri_push_max) * (-weight)
            elif weight > 0:  # 表示相吸，距离越远分越低，使用下弯曲线
                score_distance = curve_downward_2parm(x=distance, x_min=cri_pull_min, x_max=cri_pull_max) * (weight)
        else:
            score_distance = 0
        return score_distance

    room_names_exist = gpd_rooms.columns
    room_names_exist1 = gpd_rooms.columns.tolist() + ['entrance']
    # print(room_names_exist1)
    for col in df_weight_absolute.columns:
        if col in room_names_exist1:  # 确保col房间存在
            se_sub = df_weight_absolute.loc[:, col]
            se_sub = se_sub.dropna()
            score_col_total = 0
            num_col_related = 0
            for id in se_sub.index:
                if (id == 'bath') and (df_weight_absolute.loc[id, col] > 0):  # 如果是卫生间，且为吸引力，只需要有一个卫生间满足就行
                    dic_compare = {}
                    for i in room_names_exist1:
                        if i.startswith('bath'):
                            if col.startswith('room') and i.endswith('sub') and (i[-5] == col[-1]):  # 如果卧室存在独立卫生间，则该卧室直接记最高分
                                score_distance = 1 * abs(df_weight_absolute.loc[id, col])
                                dic_compare[i] = score_distance
                            elif i.startswith('bath') and not i.endswith('sub'):  # 计算剩余普通卫生间（不包括独卫）分数
                                score_distance = calculate_score_distance(col=col, id=i, name_replace=id)
                                dic_compare[i] = score_distance
                    if dic_compare:  # 排除 dic_compare为空集情况，即id不存在的情况
                        max_key = max(dic_compare, key=dic_compare.get)
                        score_col_total += dic_compare[max_key]
                        num_col_related += 1
                else:
                    if id in room_names_exist1:  # 确保id房间存在
                        # 筛选出graph中起点和终点为指定名称的行
                        score_distance = calculate_score_distance(col=col, id=id)
                        score_col_total += score_distance
                        num_col_related += 1
            if num_col_related != 0:
                dic_score[col] = score_col_total / num_col_related
    # print(dic_score)
    score_total = sum(dic_score.values())
    return score_total, dic_score, distance_max

# 上扬正弦曲线
def curve_upward_2parm(x, x_min, x_max):
    if x <= x_min:
        score = 0
    elif x >= x_max:
        score = 1
    else:
        x_std = (np.pi/2) * ((x - x_min) / (x_max - x_min))
        score = np.sin(x_std)
    return score

# 下弯余弦曲线
def curve_downward_2parm(x, x_min, x_max):
    if x <= x_min:
        score = 1
    elif x >= x_max:
        score = 0
    else:
        x_std = (np.pi/2) * ((x - x_min) / (x_max - x_min))
        score = np.cos(x_std)
    return score

# 定义一个递归函数
def custom_product_dict(d, prefix=None):
    # 如果前缀为空，则初始化为一个空列表
    if prefix is None:
        prefix = []
    # 如果字典为空，返回包含单个前缀的列表
    if not d:
        return [prefix]
    # 选择字典中的第一个键值对
    key, next_values = next(iter(d.items()))
    # 从字典中移除这个键值对
    rest_dict = {k: v for k, v in d.items() if k != key}
    # 递归调用函数，并为每个值添加键
    result = []
    for values in next_values:
        for value in values:
            # 确保直接添加坐标元组而不是列表
            # new_prefix = list(prefix)  # 创建前缀的副本以避免在迭代中修改
            # new_prefix.append((key, value))  # 添加当前键和值（坐标元组）
            # result.extend(custom_product_dict(rest_dict, new_prefix))
            result.extend(custom_product_dict(rest_dict, prefix + [(key, value)]))
    return result


######临时添加房间入口
def find_shortest_overlap_midpoints(pub, rooms, rooms_name):
    """
    给定一个多边形 pub 和一组多边形 rooms，找出每个 room 与 pub 边界重合的最短边界线的中点坐标。
    重合规则：部分重合也算重合，只有点重合不算重合。

    :param pub: 多边形对象，表示公共区域。
    :param rooms: 多边形对象列表，表示房间区域。
    :param rooms_name: 房间的名称列表。
    :return: 每个房间名称及其最短重合边界的中点坐标列表。
    """
    # 将 pub 多边形的边界转换为 LineString 集合
    pub_boundary_segments = [LineString([pub.boundary.coords[i], pub.boundary.coords[i + 1]])
                             for i in range(len(pub.boundary.coords) - 1)]

    # 计算每个 room 的重合边界
    shortest_overlap = []
    for room, name in zip(rooms, rooms_name):
        # 将 room 边界转换为 LineString 集合
        room_boundary_segments = [LineString([room.boundary.coords[i], room.boundary.coords[i + 1]])
                                  for i in range(len(room.boundary.coords) - 1)]

        # 查找重合边界
        overlap_segments = []
        for segment in room_boundary_segments:
            for pub_segment in pub_boundary_segments:
                if segment.intersects(pub_segment) and not segment.touches(pub_segment):
                    # 取两者之间更短的那一条线
                    overlap_segments.append(min(segment, pub_segment, key=lambda x: x.length))

        # 找出最短的重合线段并计算中点
        if overlap_segments:
            shortest_segment = min(overlap_segments, key=lambda x: x.length)
            midpoint = shortest_segment.interpolate(0.5, normalized=True)
            shortest_overlap.append((name, midpoint))

    # 返回每个 room 的最短重合边界的中点坐标
    return [(name, (point.x, point.y)) for name, point in shortest_overlap]


def find_adjacent_edges(outline, key_room, adjacent_preferences):
    """
    找出目标房间与餐厅、过道、客厅等交通空间的邻接关系，并提取邻接边。
    优先级示例：餐厅 > 过道 > 客厅。
    一旦存在高优先级的邻接边且长度大于800（可开门），则放弃寻找其他邻接边。
    """
    key_line = LineString(outline[key_room])  # 需要判定的房间外轮廓多段线
    # 查找邻接边
    adjacent_edges = []  # 存储超过800的相邻边
    adjacent_edges_sub = []  # 存储小于800的相邻边
    total_length = 0  # 相邻边总长度

    for pref in adjacent_preferences:
        # 如果找到邻接边，提前结束循环
        if key_room not in ['living', 'dining', 'hallway']:  # 客厅餐厅四个面都可以开门
            if not pref.startswith('passway'):
                if adjacent_edges:
                    break
        # 寻找邻接边
        adj_polygon = Polygon(outline[pref])

        # 得到房间外轮廓linestring与过道polygon的交集
        adja_i = key_line.intersection(adj_polygon)

        # 检查对象是否为MultiLineString
        if isinstance(adja_i, MultiLineString):
            # 如果是MultiLineString，分解为LineString对象
            for line in adja_i.geoms:
                lis_line_sub = list(line.coords)
                lis_lines = [LineString([lis_line_sub[i], lis_line_sub[i + 1]]) for i in range(len(lis_line_sub) - 1)]
                for line_sub in lis_lines:
                    if line_sub.length >= 800:
                        adjacent_edges.append(line_sub)
                    elif (line_sub.length < 800) and (line_sub.length > 0):
                        adjacent_edges_sub.append(line_sub)
                        total_length += line_sub.length

        elif isinstance(adja_i, LineString):
            # 如果是LineString，直接处理
            lis_lines = list(adja_i.coords)
            lis_lines = [LineString([lis_lines[i], lis_lines[i + 1]]) for i in range(len(lis_lines) - 1)]
            for line_sub in lis_lines:
                if line_sub.length >= 800:
                    adjacent_edges.append(line_sub)
                elif (line_sub.length < 800) and (line_sub.length > 0):
                    adjacent_edges_sub.append(line_sub)
                    total_length += line_sub.length

    # 存储唯一线段的列表
    unique_adjacent_edges = []
    # 检查每条线段是否唯一
    for line in adjacent_edges:
        # 检查线段或其反向是否已存在于唯一线段列表中
        if not any(line.equals(existing) or line.equals(LineString(existing.coords[::-1])) for existing in
                   unique_adjacent_edges):
            unique_adjacent_edges.append(line)
    if (not unique_adjacent_edges) and total_length >= 800:
        return [adjacent_edges_sub[0]]
    return unique_adjacent_edges


def extract_points_from_linestrings(line_strings, distance=400):
    """
    从一系列LineString对象中提取距离每个端点特定距离的点。
    返回包含提取点的列表。
    """
    points = []  # 存储提取的点

    # 遍历LineString对象
    for line in line_strings:
        if line.length > 2 * distance:
            # 从起点开始插值获取点
            start_point = line.interpolate(distance)
            # 从终点开始插值获取点
            end_point = line.interpolate(line.length - distance)

            # 将点添加到列表中
            points.append(start_point)
            points.append(end_point)

    return points


def get_midpoint(line):
    """
    计算并返回给定LineString的中点。
    """
    # 获取线段的总长度
    length = line.length
    # 计算并返回中点
    return line.interpolate(length / 2)


#######创建graph的边
def create_graph_edges(pub, midpoints, points):
    """
    创建一个图的边集合，包含从多边形 pub 的边界和给定点集合生成的 LineString。

    :param pub: 多边形对象，表示公共区域。
    :param midpoints: 房间中点坐标列表。
    :param points: pub多边形的拐点坐标列表。
    :return: LineString 对象的列表，表示图的边。
    """

    # 定义多边形轮廓上的点：房间入口点 + pub 多边形的拐点
    points_on_boundary = midpoints + points
    # 将多边形的边界转换为线段集合

    # 创建图的边
    graph_edges = []
    for p1, p2 in itertools.combinations(points_on_boundary, 2):
        line = LineString([p1, p2])
        if line.intersection(pub).length == line.length:
            graph_edges.append((p1, p2))

    # 将边转换为 LineString 形式
    graph_edges_linestring = [LineString([p1, p2]) for p1, p2 in graph_edges]

    return graph_edges_linestring


# 正式生成graph并求解
def generate_path_dataframe(room_midpoints_coords, graph_edges_linestring, gpd_all):
    """
    生成一个 dataframe，包括起点名称、起点坐标、终点名称、终点坐标、经过的边的数量、经过的边的长度之和以及经过的边的多段线。

    :param room_midpoints_coords: 点集合中的每个点的名称及其坐标。
    :param pubroom_centroid_coords: 客厅餐厅中心点坐标。
    :param graph_edges_linestring: LineString 对象列表，表示图中的边。
    :return: 包含路径信息的 pandas DataFrame。
    """

    # 创建图
    G = nx.Graph()
    for edge in graph_edges_linestring:
        start, end = list(edge.coords)
        G.add_edge(start, end, weight=edge.length, line=edge)

    # 计算路径信息
    paths = []
    for (start_name, start_coord), (end_name, end_coord) in itertools.permutations(
            room_midpoints_coords, 2):
        # 如果需要区分顺序，即认为(A, B)和(B, A)是不同的组合，应使用itertools.permutations 以生成所有可能的排列，考虑元素顺序。
        # 如果不需要区分顺序，则使用itertools.combinations。
        try:
            # 计算最短路径
            shortest_path_edges = nx.shortest_path(G, source=start_coord, target=end_coord, weight='weight')
            path_edges = [G.edges[shortest_path_edges[i], shortest_path_edges[i + 1]]['line']
                          for i in range(len(shortest_path_edges) - 1)]

            if start_name in ['living', 'dining']: # 得到减去起点房间轮廓的路径
                edges_actual = []
                polygon_room = gpd_all.loc['poly', start_name]
                for edge in path_edges:
                    edge1 = edge.difference(polygon_room)
                    edges_actual.append(edge1.length)
                path_length = sum(edges_actual)
            elif end_name in ['living', 'dining']:
                edges_actual = []
                polygon_room = gpd_all.loc['poly', end_name]
                for edge in path_edges:
                    edge1 = edge.difference(polygon_room)
                    edges_actual.append(edge1.length)
                path_length = sum(edges_actual)
            else:
                path_length = sum(edge.length for edge in path_edges)

            # 记录路径信息
            paths.append({
                "Start Name": start_name,
                "Start Coord": start_coord,
                "End Name": end_name,
                "End Coord": end_coord,
                "Number of Segments": len(path_edges),
                "Total Length": path_length,
                "Edges": path_edges
            })
        except nx.NetworkXNoPath:
            # 如果没有路径，则忽略
            continue
        except nx.NodeNotFound:
            # 处理节点不存在的情况
            # print("One of the nodes not found in the graph.")
            continue

    # 转换为 pandas DataFrame
    df_paths = pd.DataFrame(paths)
    return df_paths

# 先对df_paths进行预处理，然后用plot_shapes函数画出来
def preprocess_df_paths_for_plot(df_paths):
    """
    预处理 df_paths DataFrame，提取出多边形、线和点的集合供绘图使用。

    :param df_paths: 包含路径信息的 pandas DataFrame。
    :return: 三个列表：多边形、线和点。
    """
    lines = [edge for row in df_paths['Edges'] for edge in row]
    points = list(set(df_paths['Start Coord'].tolist() + df_paths['End Coord'].tolist()))

    # 这里假设没有多边形
    polygons = None

    return polygons, lines, points


######
######评分函数
# 查找深度和距离的函数
# 返回（经过的分段数，距离）
def find_relation(df_paths, start, end):
    # 从起点到终点
    direct_path = df_paths[((df_paths['Start Name'] == start) & (df_paths['End Name'] == end))]


    if not direct_path.empty:
        return direct_path['Number of Segments'].values[0], direct_path['Total Length'].values[0]
    # 如果没有直接路径，尝试从终点到起点
    reverse_path = df_paths[((df_paths['Start Name'] == end) & (df_paths['End Name'] == start))]
    if not reverse_path.empty:
        return reverse_path['Number of Segments'].values[0], reverse_path['Total Length'].values[0]
    return None, None  # 如果没有找到路径



# def find_edge_with_point(coords, point):
#     """
#     给定一个多边形和一个点，找出点所在的多边形边。
#     返回:
#     - LineString对象，表示点所在的多边形边。
#       如果点不在任何边上，返回 None。
#     """
#     # 获取多边形的坐标序列（外环）
#     # coords = list(polygon.exterior.coords)
#     # 遍历多边形的每条边
#     for i in range(len(coords) - 1):
#         # 创建当前边的LineString对象
#         edge = LineString([coords[i], coords[i + 1]])
#
#         # 检查给定点是否在这条边上
#         if edge.contains(Point(point)):
#             return edge  # 返回这条边
#     # 如果没有找到点所在的边，返回None
#     return None

# 从给定的起始线段（房间内的一个边界）的中点，创建一条指向房间外部的射线
# 制作一个由房间轮廓线上的一段线段向房间外发射的射线
# def make_outside_ray(start_line, room_shape):
#     center_point = list(start_line.centroid.coords)[0]
#     end_points = list(start_line.coords)
#     a = [center_point[0], center_point[1]]
#     b = [center_point[0], center_point[1]]
#     ray_point = None
#
#     for idx, item in enumerate(end_points[0]):
#         if item == end_points[1][idx]:
#             a[idx] = a[idx] + 1
#             b[idx] = b[idx] - 1
#             if Point(a).within(Polygon(room_shape)):
#                 b[idx] = b[idx] - 12000
#                 ray_point = b
#             else:
#                 a[idx] = a[idx] + 12000
#                 ray_point = a
#
#     ray_line = LineString([center_point, ray_point])
#     return ray_line


# def check_view_relationship_score(room_door_points_dict, spaces, pub, room_exist):
#     # 初始化总得分
#     total_score = 0
#     toilet_edge = find_edge_with_point(spaces['bath1'], room_door_points_dict['bath1'])
#
#     if 'dining' in room_exist:
#         # 条件1: 从卫生间入口中点向外作垂线，判断是否与【餐厅】任意位置相交
#         if make_outside_ray(toilet_edge, spaces['bath1']).intersects(Polygon(spaces['dining'])):
#             total_score -= 5
#             # print('1')
#
#     if 'living' in room_exist:
#         # 条件2: 从卫生间入口中点向外作垂线，判断是否与【客厅】任意位置相交
#         if make_outside_ray(toilet_edge, spaces['bath1']).intersects(Polygon(spaces['living'])):
#             total_score -= 4
#             # print('2')
#
#     # 条件3: 将卫生间入口与大门连线，判断是否视线可达
#     toi_ent_line = LineString([room_door_points_dict['entrance'], room_door_points_dict['bath1']])
#     if toi_ent_line.intersection(pub).equals(toi_ent_line):
#         total_score -= 3
#         # print('3')
#
#     # 条件4: 将卧室入口与大门连线，判断是否视线可达
#     for room in room_exist:
#         if room.startswith('room'):
#             rooms_ent_line = LineString([room_door_points_dict['entrance'], room_door_points_dict[room]])
#             if rooms_ent_line.intersection(pub).equals(rooms_ent_line):
#                 total_score -= 2
#                 # print('4')
#     return total_score


def plot_shapes(polygons=None, lines=None, points=None):
    """
    绘制多边形、线的集合和点的集合。

    :param polygons: 多边形对象的列表。
    :param lines: 线对象的列表。
    :param points: 点坐标的列表。
    """
    import matplotlib.pyplot as plt
    from shapely.geometry import LineString

    plt.figure(figsize=(10, 10))

    # 绘制多边形
    if polygons is not None:
        for polygon in polygons:
            x, y = polygon.exterior.xy
            plt.fill(x, y, alpha=0.3)  # 透明度填充

    # 绘制线
    if lines is not None:
        for line in lines:
            x, y = line.xy
            plt.plot(x, y)

    # 绘制点
    if points is not None:
        x, y = zip(*points)  # 解包点坐标
        plt.scatter(x, y, marker='o')  # 绘制点

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-1000, 15000)
    plt.ylim(-1000, 15000)
    plt.title('Shapes Plot')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# 示例用法
# poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
# line = LineString([(2, 0), (3, 1)])
# pts = [(4, 4), (5, 5)]

# plot_shapes(polygons=[pub], lines=[line], points=pts)


if __name__ == "__main__":
    import os
    pd.set_option('display.max_columns', None)
    # path_cri = r'criterion_setting_file\shape_rooms.xlsx'
    path_cri = 'criterion_setting_file/shape_rooms.xlsx'
    df_criterion = pd.read_excel(path_cri, index_col=0)

    wrong_file_list = []
    count = 0
    circulation_absolute = pd.read_excel('criterion_setting_file/relation_circulation_absolute.xlsx', index_col=0)
    circulation_relative = pd.read_excel('criterion_setting_file/relation_circulation_relative.xlsx', index_col=0)
    path = '../cases_for_test_large/2_improved/'

    recorder = []
    for root, dirs, files in os.walk(path):
        print("file nums:", len(files))
        for id, file in enumerate(files):
            if file.endswith('xlsx') and (id == 0):
                # if file.endswith('xlsx'):
                # if file == '1layout_info.xlsx':
                    count += 1
                    print(file)
                    df = pd.read_excel(path + file,
                                       # sheet_name='room_info',
                                       index_col=0)
                    df = df.rename(columns={'dinner': 'dining', 'toilet1': 'bath1', 'toilet_main':'bath1'})
                    # print(df)
                    geo_layout = layout2geopandas(layout_info=df)
                    adj = adjacent_matrix_shapely(df_shapely=geo_layout)

                    case = RoomShapeScore(df_shape=geo_layout, df_info=df, adj_matrix=adj, criterion=df_criterion)
                    out = case.room_shape_total_score()  # 需要先跑房间得分，才能进一步明确房间形状
                    df_shape = case.df_shape
                    geo_layout[df_shape.columns] = df_shape.values
                    score = circulation_evaluator(
                        gpd_rooms=df_shape,
                        gpd_all=geo_layout,
                        df_weight_absolute=circulation_absolute,
                        df_weight_relative=circulation_relative
                    )
                    print(score)
                    recorder.append(score)
                    if score == -100:
                        wrong_file_list.append(file)
                    print('--------------')

    # print('得分统计：', min(recorder), max(recorder))
    for i in wrong_file_list:
        print(i)

























