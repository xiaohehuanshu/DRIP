import numpy as np
import pandas as pd
import shapely
from tools_layout_modeling import layout2geopandas, adjacent_matrix_shapely
from utils.graph_layout import GraphPlan
from shapely.geometry import Point, LineString
from shapely.ops import unary_union

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

'''
房间采光和朝向评分, 包括 '朝向', '采光宽度', '采光遮挡?'
'''

def normalization_room_light(func):
    def wrapper(*args, **kwargs):
        # 在函数调用前执行额外的操作
        result = func(*args, **kwargs)
        # 在函数调用后执行额外的操作
        total_score = result / 2.65
        return total_score
    return wrapper


class RoomLightScore:
    def __init__(self, df_shape, df_info, criterion, criterion_win, room_names_need):
        self.df_shape = df_shape
        self.df_info = df_info
        self.param = 30000
        self.rooms_need = [
            'room1', 'room2', 'room3', 'room4', 'living', 'dining', 'kitchen', 'study_room', 'staircase',
            'bath1', 'bath2', 'bath1_sub', 'hallway', 'storeroom', 'garage'
        ]
        self.whites_main = ['white_south', 'white_north', 'white_west', 'white_east']
        self.whites_valid = []
        self.whites_main_valid = []
        self.whites_sub_valid = []
        for i in df_shape.columns:
            if 'white' in i:
                self.whites_valid.append(i)
                if i in self.whites_main:
                    self.whites_main_valid.append(i)
                if i[-2] == 'm':
                    self.whites_sub_valid.append(i)

        self.rooms_valid = list(set(df_shape.columns).intersection(set(self.rooms_need)))
        self.room_names_need = room_names_need
        # for i in self.whites_main:
        #     if i in df_shape.columns:
        #         self.rooms_valid.append(i)

        self.criterion = criterion
        self.criterion_win = criterion_win.dropna(axis=1, how='any')

        self.room_shapes_union = unary_union(self.df_shape[self.rooms_valid].loc['poly'])
        self.white_m_union = unary_union(self.df_shape[self.whites_sub_valid].loc['poly'])
        self.white_all_union = unary_union(self.df_shape[self.whites_sub_valid + self.whites_main_valid].loc['poly'])

        self.light_scores = pd.DataFrame(columns=self.rooms_valid, index=['orientation', 'win_width'])

        poly_boundary = self.df_shape.loc["rec", "boundary"]
        env_other_room_names = [i for i in self.df_shape.columns if i.startswith('white')]
        poly_other_env = self.df_shape.loc["rec", env_other_room_names]
        poly_other_env_unit = unary_union(poly_other_env)
        self.poly_blank_env = poly_boundary.difference(poly_other_env_unit)  # 室内空间区域

    @normalization_room_light
    def room_light_total_score(self):
        for i in self.rooms_valid:
            score_ori, score_win, name_ori = self.score_light(room_name=i)
            score_orientation = score_ori * 3
            self.light_scores[i] = [score_orientation, score_win]
        # print(self.light_scores)
        total_score = self.light_scores.values.sum() / len(self.room_names_need)  # 计算单个房间平均得分
        return total_score

    def score_light(self, room_name):
        '''每个房间与不同白体之间都有权重分值，如果该房间满足了与某个白体的朝向关系，则获得该权重的得分。多个得分取最高分'''
        '''按照射线法，即从房间向四个方向发射射线，并检测是否被遮挡'''
        '''同时计算各个房间最大开窗尺寸得分'''

        dic_window_lines = {}  # 用于盛放与房间i共线的所有白体及对应的线段
        lis_white_orientation = []

        dic_lines = self.make_ray_four_directions(room_name)  # 得到该房间四个方向的射线

        for key in dic_lines.keys():
            light_ray = dic_lines[key][0]  # 房间射线
            light_edge_raw = dic_lines[key][1]  # 房间边缘线

            rooms_valid = self.rooms_valid.copy()
            rooms_valid.remove(room_name)
            rooms_union_sub = unary_union(self.df_shape.loc['rec', rooms_valid])
            light_edge = light_edge_raw.difference(rooms_union_sub)  # 减去被其它房间覆盖的部分

            if light_ray.intersection(self.poly_blank_env).length < 0.01:  # 如果射线不与boundary内的空白区域相交，则该窗子直接对外
                if key in self.df_shape.columns:  # 判断该方向的白体存在
                    lis_white_orientation.append(key)
                    line_share_length = self.white_all_union.intersection(light_edge).length
                    dic_window_lines[key] = line_share_length

        # 比较所有朝向的优先级, 并根据评分表打分
        if len(lis_white_orientation) != 0:
            sub_criterion = self.criterion.loc[room_name]
            orientations_potential = sub_criterion[lis_white_orientation]  # 筛选实际有朝向信息的白体
            orientations_potential = orientations_potential.sort_values(ascending=False)  # 将朝向评分表中的白体按权重排序
            final_orientation_name = orientations_potential.index[0]
            final_orientation_score = sub_criterion[final_orientation_name] / self.criterion.values.max()
            final_window_score = self.score_light_width(window_length=dic_window_lines[final_orientation_name],
                                                        room_name=room_name)
            # print(room_name, final_orientation_name)

        else:  # 无主白体朝向，但与white_m相邻
            final_orientation_score = 0
            final_orientation_name = None
            final_window_score = self.score_light_width(window_length=sum(dic_window_lines.values()),
                                                        room_name=room_name)
        # print(room_name, final_window_score, final_orientation_score)
        return final_orientation_score, final_window_score, final_orientation_name  # 朝向得分，最终窗线信息，最终朝向房间名称

    def score_light_width(self, window_length, room_name):
        if (room_name in self.criterion_win.columns) and window_length:
            score = self.curve_2parm_upwards(x=window_length,
                                             x_min=self.criterion_win.loc['width_min', room_name],
                                             x_max=self.criterion_win.loc['width_optim_min', room_name])
            if (room_name == 'room1') and ('bath1_sub' in self.df_info.columns):  # 主卫情况特殊考虑，尽量让卫生间在内测
                score = score * 2
        else:
            score = 0
        return score

    # 光线遮挡得分，目前朝向的判定中已经考虑了采光遮挡因素（如果中心射线被遮挡，只计算采光宽度分数），因此可不计算采光遮挡。
    def score_light_block(self, window_line, room_name, white_orientation):
        if white_orientation is None:
            score = 0
            return score
        out_rays = self.make_outside_ray3(start_line=window_line, room_shape=self.df_shape.loc['poly', room_name])
        counter = 0
        for ray in out_rays:
            if not ray.crosses(self.room_shapes_union):
                if ray.intersects(self.df_shape.loc['poly', white_orientation]):
                    counter += 1
        score = counter / 3
        return score

    def make_ray_four_directions(self, room_name):
        '''得到从某个房间四个边发出的四条射线'''

        x, y, w, d = self.df_info.loc[:, room_name]
        ray_start_south = [x + w / 2, y]
        ray_end_south = [x + w / 2, y - self.param]
        line_south = shapely.LineString([ray_start_south, ray_end_south])  # 房间边缘为起点的射线
        line_south_room = shapely.LineString([(x, y), (x+w, y)])  # 房间边缘线

        ray_start_north = [x + w / 2, y + d]
        ray_end_north = [x + w / 2, y + d + self.param]
        line_north = shapely.LineString([ray_start_north, ray_end_north])
        line_north_room = shapely.LineString([(x, y+d), (x+w, y+d)])

        ray_start_west = [x, y + d / 2]
        ray_end_west = [x - self.param, y + d / 2]
        line_west = shapely.LineString([ray_start_west, ray_end_west])
        line_west_room = shapely.LineString([(x, y), (x, y+d)])

        ray_start_east = [x + w, y + d / 2]
        ray_end_east = [x + w + self.param, y + d / 2]
        line_east = shapely.LineString([ray_start_east, ray_end_east])
        line_east_room = shapely.LineString([(x+w, y), (x+w, y+d)])

        dic = {
            'white_south': [line_south, line_south_room],  # 房间边缘为起点的射线, 房间边缘线
            'white_north': [line_north, line_north_room],
            'white_west': [line_west, line_west_room],
            'white_east': [line_east, line_east_room]
        }
        return dic

    # 制造一个由房间外轮廓线上的一段线段向房间外发射的射线
    def make_outside_ray1(self, start_line, room_shape):
        '''向外部发射一条射线'''
        center_point = [start_line.centroid.x, start_line.centroid.y]
        end_points = list(start_line.coords)
        a = [center_point[0], center_point[1]]
        b = [center_point[0], center_point[1]]
        ray_point = center_point.copy()

        for idx, item in enumerate(end_points[0]):
            if item == end_points[1][idx]:
                a[idx] = a[idx] + 1
                b[idx] = b[idx] - 1
                if Point(a).within(room_shape):
                    ray_point[idx] = ray_point[idx] - 6000
                else:
                    ray_point[idx] = ray_point[idx] + 6000
        ray_line_mid = LineString([center_point, ray_point])
        return ray_line_mid

    def make_outside_ray3(self, start_line, room_shape):
        '''向外部发射三条射线'''
        center_point = [start_line.centroid.x, start_line.centroid.y]
        end_points = list(start_line.coords)
        a = [center_point[0], center_point[1]]
        b = [center_point[0], center_point[1]]
        ray_point = center_point.copy()
        ray_point0 = list(end_points[0])
        ray_point1 = list(end_points[1])

        for idx, item in enumerate(end_points[0]):
            if item == end_points[1][idx]:
                a[idx] = a[idx] + 1
                b[idx] = b[idx] - 1
                if Point(a).within(room_shape):
                    ray_point[idx] = ray_point[idx] - 6000
                    ray_point0[idx] = ray_point0[idx] - 6000
                    ray_point1[idx] = ray_point1[idx] - 6000

                else:
                    ray_point[idx] = ray_point[idx] + 6000
                    ray_point0[idx] = ray_point0[idx] + 6000
                    ray_point1[idx] = ray_point1[idx] + 6000

        ray_line_mid = LineString([center_point, ray_point])
        ray_line_end0 = LineString([end_points[0], ray_point0])
        ray_line_end1 = LineString([end_points[1], ray_point1])
        return ray_line_mid, ray_line_end0, ray_line_end1

    def curve_2parm_upwards(self, x, x_min, x_max):
        if x < x_min:
            score = 0
        elif (x >= x_min) and (x < x_max):
            x_std = (np.pi / 2) * ((x - x_min) / (x_max - x_min))
            score = np.sin(x_std)
        else:
            score = 1
        return score


# if __name__ == '__main__':
#     import os
#     path_c = 'D:/ON_GOING/RL_house/CODE/RL_evaluator/criterion_setting_file/relation_light.xlsx'
#     path_win = 'criterion_setting_file/shape_windows.xlsx'
#     df_criterion = pd.read_excel(path_c, index_col=0)
#     df_criterion_win = pd.read_excel(path_win, index_col=0)
#     df1 = df_criterion_win.dropna(axis=1, how='any')

#     path = '../cases_for_test_large/2_improved/'
#     # path = 'D:/ON_GOING/RL_house/CODE/mcts_layout-master/cache/compare/'
#     recorder = []
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             if file.endswith('xlsx'):
#                 print(file)
#                 df = pd.read_excel(path+file, index_col=0)
#                 df = df.rename(columns= {'dinner': 'dining', 'toilet_main': 'bath1'})
#                 rooms_names_need = ['room1', 'room2', 'room3', 'room4', 'living', 'dining', 'kitchen',
#                                     'bath1', 'bath2', 'bath1_sub', 'bath2_sub', 'hallway', 'storeroom']
#                 rooms_names_need = list(set(rooms_names_need).intersection(set(df.columns)))
#                 geo_layout = layout2geopandas(layout_info=df)
#                 adj = adjacent_matrix_shapely(df_shapely=geo_layout)
#                 case = RoomLightScore(df_shape=geo_layout, adj_matrix=adj, criterion=df_criterion,
#                                       criterion_win=df_criterion_win, room_names_need=rooms_names_need)
#                 out = case.room_light_total_score()
#                 recorder.append(out)
#                 print(out)
#                 print('--------------')
#     print('得分统计：', min(recorder), max(recorder))


    # path_c = 'D:/ON_GOING/RL_house/CODE/RL_evaluator/criterion_setting_file/relation_light.xlsx'
    # path_win = 'D:/ON_GOING/RL_house/CODE/RL_evaluator/criterion_setting_file/shape_windows.xlsx'
    # df_criterion = pd.read_excel(path_c, index_col=0)
    # df_criterion_win = pd.read_excel(path_win, index_col=0)
    # df1 = df_criterion_win.dropna(axis=1, how='any')
    #
    # path = '../cases_for_test_large/2/A-08.xlsx'
    # df = pd.read_excel(path, index_col=0)
    # df = df.rename(columns= {'dinner': 'dining', 'toilet_main': 'bath1'})
    #
    # geo_layout = layout2geopandas(layout_info=df)
    # adj = adjacent_matrix_shapely(df_shapely=geo_layout)
    # case = RoomLightScore(df_shape=geo_layout, adj_matrix=adj, criterion=df_criterion, criterion_win=df_criterion_win)
    #
    # out = case.room_light_total_score()
    # print(out)








