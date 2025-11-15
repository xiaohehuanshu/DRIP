import numpy as np
import pandas as pd
from tools_layout_modeling import layout2geopandas, adjacent_matrix_shapely
from .tools_space_analysis import clean_polygon_midpoints
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import split
import shapely

'''
房间形式评分, 包括: 'area', 'ratio', 'earea', 'eratio', 'regular_shape'
'''

def normalization_room_shape(func):
    def wrapper(*args, **kwargs):
        # 在函数调用前执行额外的操作
        result = func(*args, **kwargs)
        # 在函数调用后执行额外的操作
        # result.loc['regular_shape', :] = result.loc['regular_shape', :] * 0.3
        # total_score = result.iloc[:4, :]
        total_score = result
        total_score = total_score - 0.16
        total_score = total_score / 1.84
        return total_score
    return wrapper


class RoomShapeScore:
    def __init__(self, df_shape, df_info, adj_matrix, criterion):
        self.rooms_need = [
            'room1', 'room2', 'room3', 'room4', 'living', 'dining', 'kitchen', 'study_room', 'staircase',
            'bath1', 'bath2', 'bath1_sub', 'hallway', 'storeroom', 'garage'
        ]
        names = [i for i in self.rooms_need if i in df_info.columns]
        self.df_shape = df_shape[names].copy()
        self.df_info = df_info[names].copy()
        self.adj_matrix = adj_matrix.loc[names, names].copy()
        self.criterion = criterion
        self.df_info.index = ['x', 'y', 'w', 'd']

        self.room_scores = pd.DataFrame(columns=self.df_info.columns, index=['area', 'ratio'])
        self.rooms_with_intersection = []

    # 计算房间得分的总函数
    @normalization_room_shape
    def room_shape_total_score(self):
        # 先用附属房间减去主房间，再做其它计算
        if ('room1' in self.df_shape.columns) and ('bath1_sub' in self.df_shape.columns):
            room1_diff = self.df_shape.loc['rec', 'room1'].difference(self.df_shape.loc['rec', 'bath1_sub'])
            self.df_shape.loc['poly', 'room1'] = room1_diff

        # 查找有交集的房间，并判断交集归属
        recorder_scored_rooms = []  # 记录已经被打过分的房间
        for i, idx in enumerate(self.df_shape.columns):
            idx_sub = self.adj_matrix.loc[idx][i:]
            if 2 in idx_sub.values:
                idx_sub = idx_sub[idx_sub == 2]
                # 遍历有交集的房间
                for col in idx_sub.index:
                    room_idx_shape = self.df_shape.loc['poly', idx]
                    room_col_shape = self.df_shape.loc['poly', col]
                    room_idx_info = self.df_info[idx]
                    room_col_info = self.df_info[col]

                    # 判定交集的归属问题
                    room_idx_diff = room_idx_shape.difference(room_col_shape)
                    room_col_diff = room_col_shape.difference(room_idx_shape)

                    score_idx = self.decide_room_score(room_name=idx, room_shape=room_idx_shape, room_info=room_idx_info)
                    score_col = self.decide_room_score(room_name=col, room_shape=room_col_shape, room_info=room_col_info)
                    score_idx_diff = self.decide_room_score(room_name=idx, room_shape=room_idx_diff, room_info=room_idx_info)
                    score_col_diff = self.decide_room_score(room_name=col, room_shape=room_col_diff, room_info=room_col_info)
                    # if idx == 'kitchen':
                    #     print(sum(score_idx + score_col_diff))
                    #     print(sum(score_col + score_idx_diff))
                    # 在self.df_shape中更新剪裁的房间
                    # 在self.rooms_with_intersection中记录被剪裁的房间名称
                    # 在self.room_scores中更新个房间得分
                    if (sum(score_idx) + sum(score_col_diff)) < (sum(score_col) + sum(score_idx_diff)):
                        self.df_shape.loc['poly', idx] = room_idx_diff
                        self.rooms_with_intersection.append(idx)
                        self.room_scores[idx] = score_idx_diff
                        self.room_scores[col] = score_col
                        recorder_scored_rooms.append(idx)
                        recorder_scored_rooms.append(col)

                    else:
                        self.df_shape.loc['poly', col] = room_col_diff
                        self.rooms_with_intersection.append(col)
                        self.room_scores[col] = score_col_diff
                        self.room_scores[idx] = score_idx
                        recorder_scored_rooms.append(idx)
                        recorder_scored_rooms.append(col)

                    # print(idx, sum(score_idx), sum(score_col_diff), sum(score_idx + score_col_diff))
                    # print(col, sum(score_col), sum(score_idx_diff), sum(score_col + score_idx_diff))
                    # print(idx, f'{col}_diff', score_idx, score_col_diff)
                    # print(col, f'{idx}_diff', score_col, score_idx_diff)
                    # print('------------')
            elif (2 not in idx_sub.values) and (idx not in recorder_scored_rooms):
                score_room_idx = self.score_rectangle_room(room_shape=self.df_shape.loc['rec', idx], room_name=idx)
                self.room_scores[idx] = score_room_idx
            else:
                # print(idx)
                pass
        # print(self.room_scores)
        additional_score = self.decide_additional_score()  # 嵌套房间的内部关系得分
        if len(self.df_shape.columns) == 0:
            return 0
        # print(self.room_scores)
        total_score = self.room_scores.values.sum() / len(self.df_shape.columns)
        total_score = additional_score / len(self.df_shape.columns) + total_score

        return total_score


    # 得到房间中的有效空间（最大内切矩形）
    def get_effective_subrectangle(self, poly_room, rec_room):
        rectangle_effective = None
        rectangle_area_counter = 0
        if poly_room.geom_type == 'MultiPolygon':
            return None

        # 切分凹多边形为4个小矩形
        list_poly = list(poly_room.exterior.coords)
        for i, p in enumerate(list_poly):
            if not Point(p).within(rec_room.exterior):
                point1 = Point(p[0], p[1] + 12000)
                point1_ = Point(p[0], p[1] - 12000)
                line1 = shapely.LineString([point1, point1_])
                recs_split1 = split(poly_room, line1)
                for j in recs_split1.geoms:
                    if j.area > rectangle_area_counter:
                        rectangle_effective = j
                        rectangle_area_counter = j.area

                point2 = Point(p[0] + 12000, p[1])
                point2_ = Point(p[0] - 12000, p[1])
                line2 = LineString([point2, point2_])
                recs_split2 = split(poly_room, line2)
                for m in recs_split2.geoms:
                    if m.area > rectangle_area_counter:
                        rectangle_effective = m
                        rectangle_area_counter = m.area
        return rectangle_effective

    def decide_room_score(self, room_name, room_shape, room_info):
        score_decide = None
        # 消除多边形中的非拐点
        if (room_shape.geom_type == 'Polygon') and (not room_shape.is_empty):
            room_shape_clean = clean_polygon_midpoints(shape=room_shape)
            len_vertices = len(list(room_shape_clean.exterior.coords))
            # print(room_shape_clean, room_shape_clean.is_empty)
            # 根据顶点数量判定是矩形还是多边形，进而分情况评分
            if not room_shape_clean.is_empty:
                if len_vertices <= 7:
                    if len_vertices <= 5:  # 房间不缺角
                        score_decide = self.score_rectangle_room(room_shape=room_shape_clean,
                                                                 room_name=room_name)
                    elif (len_vertices > 5) and (len_vertices <= 7):  # 房间缺一个角
                        score_decide = self.score_poly_room(room_shape=room_shape_clean, room_info=room_info,
                                                            room_name=room_name)
                else:  # 房间缺两个以上的角
                    score_decide = self.score_poly_room(room_shape=room_shape_clean, room_info=room_info,
                                                        room_name=room_name, if_common=False)
        else:
            # score_decide = [0, 0, 0, 0, 0]
            score_decide = [0, 0]
        return score_decide

    def decide_additional_score(self):
        '''对嵌套房间等的尺寸关系打分'''
        score_addition = 0
        if ('room1' in self.df_shape.columns) and ('bath1_sub' in self.df_shape.columns):
            # 如果主卫是包含在主卧中的，计算主卧的有效区域是否适合放床，以及开门处能够通过
            if self.df_shape.loc['rec', 'room1'].intersection(self.df_shape.loc['rec', 'bath1_sub']).area > 1:
                # 计算放床区域墙线长度得分
                eshape_room1 = self.get_effective_subrectangle(poly_room=self.df_shape.loc['poly', 'room1'],
                                                               rec_room=self.df_shape.loc['rec', 'room1'])
                if eshape_room1 != None:
                    shape_bath1_sub = self.df_shape.loc['rec', 'bath1_sub'].exterior
                    len_share_line = eshape_room1.intersection(shape_bath1_sub).length  # 主卧中与床平行的墙线
                    score_bed = self.curve_4parm(x=len_share_line, x_max=3300, x_min=1500, x_opt_max=2700, x_opt_min=2100) * 0.6
                    self.sub_room_score = score_bed
                    score_addition = score_bed

                    # 计算卧室最窄宽度得分（入口过道）
                    room1 = self.df_shape.loc['poly', 'room1']
                    if room1.geom_type == "Polygon":
                        room1_line = room1.exterior.coords[:]
                        length_min = 1000000
                        for i in range(len(room1_line)-1):
                            line = LineString([room1_line[i], room1_line[i+1]])
                            if line.length < length_min:
                                length_min = line.length
                        if length_min >= 900:
                            score_addition += 0.4
        return score_addition


    # 计算非矩形房间的分值，包括总体得分以及有效面积得分
    def score_poly_room(self, room_shape, room_info, room_name, if_common=True):
        if len(list(room_shape.exterior.coords)) > 7:
            # score_all = [0, 0, 0, 0, 0]
            score_all = [0, 0]
            return score_all

        width = room_info['w']
        depth = room_info['d']
        area = room_shape.area
        ratio = min([width / depth, depth / width])

        ## 外接矩形得分
        # 面积得分
        # score_area = self.curve_4parm(
        #     x=area, x_max=self.criterion.loc['area_max', room_name],
        #     x_min=self.criterion.loc['area_min', room_name],
        #     x_opt_max=self.criterion.loc['area_optim_max', room_name],
        #     x_opt_min=self.criterion.loc['area_optim_min', room_name]
        # )
        # # 长宽比得分
        # score_ratio = self.curve_3parm_optimal_ratio(
        #     x=ratio, x_min=self.criterion.loc['ratio_min', room_name],
        #     x_opt_max=self.criterion.loc['ratio_optim_max', room_name],
        #     x_opt_min=self.criterion.loc['ratio_optim_min', room_name]
        # )

        if if_common is True:
            # 有效区域得分
            eshape = self.get_effective_subrectangle(poly_room=room_shape, rec_room=self.df_shape.loc['rec', room_name])
            if eshape is not None:
                edepth = abs(eshape.bounds[3] - eshape.bounds[1])
                ewidth = abs(eshape.bounds[2] - eshape.bounds[0])
                earea = eshape.area
                eratio = min([ewidth / edepth, edepth / ewidth])

                score_area = self.curve_4parm(
                    x=earea, x_max=self.criterion.loc['area_max', room_name],
                    x_min=self.criterion.loc['area_min', room_name],
                    x_opt_max=self.criterion.loc['area_optim_max', room_name],
                    x_opt_min=self.criterion.loc['area_optim_min', room_name]
                )
                # 长宽比得分
                score_ratio = self.curve_3parm_optimal_ratio(
                    x=eratio, x_min=self.criterion.loc['ratio_min', room_name],
                    x_opt_max=self.criterion.loc['ratio_optim_max', room_name],
                    x_opt_min=self.criterion.loc['ratio_optim_min', room_name]
                )
                score_all = [score_area, score_ratio]
                # # 有效区域面积得分
                # score_earea = self.curve_2parm_upwards(x=earea,
                #                                        x_min=self.criterion.loc['earea_min', room_name],
                #                                        x_max=self.criterion.loc['area_optim_min', room_name])
                # # 有效区域长宽比得分
                # score_eratio = self.curve_3parm_optimal_ratio(
                #     x=eratio, x_min=self.criterion.loc['ratio_min', room_name],
                #     x_opt_max=self.criterion.loc['ratio_optim_max', room_name],
                #     x_opt_min=self.criterion.loc['ratio_optim_min', room_name]
                # )
                #
                # # 形式的规整性得分
                # score_regular_shape = 0.5
                # score_all = [score_area, score_ratio, score_earea, score_eratio, score_regular_shape]
            else:
                score_all = [0, 0]
        else:
            # score_all = [score_area, score_ratio, 0, 0, 0]
            score_all = [0, 0]
        return score_all

    def score_rectangle_room(self, room_shape, room_name):
        bounds = room_shape.bounds
        width = bounds[2] - bounds[0]
        depth = bounds[3] - bounds[1]
        # print(f'room name:\t{room_name}')
        # print(f'width:\t{width}')
        # print(f'depth:\t{depth}')
        if width < 1e-5 or depth < 1e-5:
            return 0
        area = width * depth
        ratio = min([width / depth, depth / width])

        # 面积得分
        score_area = self.curve_4parm(
            x=area, x_max=self.criterion.loc['area_max', room_name],
            x_min=self.criterion.loc['area_min', room_name],
            x_opt_max=self.criterion.loc['area_optim_max', room_name],
            x_opt_min=self.criterion.loc['area_optim_min', room_name]
        )
        # 长宽比得分
        score_ratio = self.curve_3parm_optimal_ratio(
            x=ratio, x_min=self.criterion.loc['ratio_min', room_name],
            x_opt_max=self.criterion.loc['ratio_optim_max', room_name],
            x_opt_min=self.criterion.loc['ratio_optim_min', room_name]
        )

        # 有效区域得分
        score_earea = 1
        score_eratio = 1

        # 形式的规整性得分
        score_regular_shape = 1

        # score_all = [score_area, score_ratio, score_earea, score_eratio, score_regular_shape]
        score_all = [score_area, score_ratio]
        return score_all

    # 四个参数评分曲线，包括最大值、最小值、最优最大值、最优最小值
    def curve_4parm(self, x, x_max, x_min, x_opt_max, x_opt_min):
        if (x > x_max) or (x < x_min):
            score = 0
        elif (x >= x_opt_min) and (x <= x_opt_max):
            score = 1
        elif (x >= x_min) and (x < x_opt_min):
            x_std = (np.pi / 2) * ((x - x_min) / (x_opt_min - x_min))
            score = np.sin(x_std)
        else:
            x_std = (np.pi / 2) * ((x - x_opt_max) / (x_max - x_opt_max)) + np.pi / 2
            score = np.sin(x_std)
        return score

    # 两个参数评分曲线，针对长宽比，包括最优最大值、最优最小值
    def curve_3parm_optimal_ratio(self, x, x_min, x_opt_max, x_opt_min):
        if (x >= x_opt_min) and (x <= x_opt_max):
            score = 1
        elif (x < x_opt_min) and (x > x_min):
            x_std = (np.pi / 2) * ((x - x_min) / (x_opt_min - x_min))
            score = np.sin(x_std)
        elif x > x_opt_max:
            x_std = (np.pi / 2) * ((x - x_opt_max) / (1 - x_opt_max)) + np.pi / 2
            score = np.sin(x_std)
        else:
            score = 0
        return score

    def curve_2parm_upwards(self, x, x_min, x_max):
        if x < x_min:
            score = 0
        elif (x >= x_min) and (x < x_max):
            x_std = (np.pi / 2) * ((x - x_min) / (x_max - x_min))
            score = np.sin(x_std)
        else:
            score = 1
        return score

    def curve_1parm_downwards_intersection_area(self, x, x_max):
        if x > x_max:
            score = 0
        else:
            score = (x_max - x) / x_max
        return score

    # 两个参数评分曲线，包括最大值、最小值
    def curve_2parm(self, x, x_max, x_min):
        if (x >= x_max) or (x <= x_min):
            score = 0
        else:
            x_std = np.pi * ((x - x_min) / (x_max - x_min))
            score = np.sin(x_std)
        return score

    def export_csv_shape_file(self, file_name):
        shapes = self.df_shape.iloc[1]
        counter = 0
        dic_shape_trans = {}
        for idx in shapes.index:
            room_coord = list(shapes[idx].exterior.coords)
            if len(room_coord) > counter:
                counter = len(room_coord)
            dic_shape_trans[idx] = room_coord
        df_coord_shapes = pd.DataFrame(columns=shapes.index, index=range(counter))
        for key in dic_shape_trans.keys():
            df_coord_shapes.loc[:len(dic_shape_trans[key]) - 1, key] = dic_shape_trans[key]
        df_coord_shapes.to_csv(f'cache/{file_name}')
        return df_coord_shapes


# if __name__ == '__main__':
#     import os
#     from utils.graph_layout import GraphPlan
#     # from utils.graph_poly_plan import GraphPolyPlan

#     # path = '../cases_for_test/case1_0.xlsx'

#     # file_path = '../cases_for_test/'
#     # file_path = '../cases_for_test_large/2_improved/'
#     # file_path = 'D:/ON_GOING/RL_house/CODE/RL_evaluator/cache/测试sub_room/'
#     file_path = 'D:/ON_GOING/RL_house/CODE/mcts_layout-master_test/cache/29-中间户型/'
#     # path_out = '../cases_for_test/'
#     path_cri = r'criterion_setting_file\shape_rooms.xlsx'
#     df_criterion = pd.read_excel(path_cri, index_col=0)
#     recorder = []

#     for root, dirs, files in os.walk(file_path):
#         for i, file in enumerate(files):
#             # if file == '1layout_info.xlsx':
#             if file[-4:] == 'xlsx' and file == '1layout_info.xlsx':
#                 print(i, file)
#                 path = file_path + file
#                 df = pd.read_excel(path, index_col=0,
#                                    # sheet_name='room_info'
#                                    )
#                 df = df.rename(columns={'dinning': 'dining', 'dinner': 'dining',
#                                         'toilet1': 'bath1', 'toilet_main': 'bath1'})

#                 geo_layout = layout2geopandas(layout_info=df)
#                 adj = adjacent_matrix_shapely(df_shapely=geo_layout)
#                 case = RoomShapeScore(df_shape=geo_layout, df_info=df, adj_matrix=adj, criterion=df_criterion)
#                 score = case.room_shape_total_score()

#                 # 得到公共区域多边形
#                 df_shape = case.df_shape
#                 geo_layout = layout2geopandas(layout_info=df)
#                 pub = get_public_space(gpd_all=geo_layout)
#                 df_shape.loc['poly', 'pub'] = pub
#                 df_shape.loc['poly', 'entrance'] = geo_layout.loc['poly', 'entrance']
#                 print(score)
#                 recorder.append(score)
#                 # plan = GraphPlan(layout_info=geo_layout, file_name=file[:-5], path_out=path_out)
#                 # plan.draw_plan()
#                 # poly_plan = GraphPolyPlan(layout_poly=df_shape, file_name=file[:-5]+'_1', path_out=file_path)
#                 # poly_plan.draw_plan()

#     # print('得分统计：', min(recorder), max(recorder))