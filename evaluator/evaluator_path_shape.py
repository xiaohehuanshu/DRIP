import time

import numpy as np
import pandas as pd
import shapely
from shapely import geometry
from shapely import ops
from tools_layout_modeling import build_room
from shapely.ops import unary_union
import matplotlib.pyplot as plt

pd.set_option('expand_frame_repr', False)


def normalization_path_shape(func):
    def wrapper(*args, **kwargs):
        # 在函数调用前执行额外的操作
        score_se, score_se_access = func(*args, **kwargs)
        # 在函数调用后执行额外的操作
        total_score = sum(score_se.values) / 5.5
        total_score_access = sum(score_se_access.values) / 2
        return total_score, total_score_access
    return wrapper


def normalization_living_dining_shape(func):
    def wrapper(*args, **kwargs):
        # 在函数调用前执行额外的操作
        result = func(*args, **kwargs)
        # print(result)
        # 在函数调用后执行额外的操作
        total_score = result
        return total_score
    return wrapper


class PathShapeScore:
    def __init__(self, df_shape):
        self.df_shape = df_shape
        self.rooms_pub = [
            'living', 'dining', 'hallway'
        ]

        self.entrances = [i for i in self.df_shape.columns if i in ['entrance', 'entrance_sub']]

        self.rooms_valid = []
        for i in self.df_shape.columns:
            if (i not in self.rooms_pub) and (i != 'boundary') and (not i.startswith('white')) and (not i.startswith('black') and (not i.startswith('entrance'))):
                self.rooms_valid.append(i)
        self.room_inner_storage = self.df_shape[self.rooms_valid].copy()  # 除boundary外的所有实体空间

        self.boundary = self.df_shape.loc['rec', 'boundary']
        self.public = None  # 包含所有公共空间的过道
        self.passway = None  # 纯过道

    @normalization_path_shape
    def path_shape_total_score(self):
        ######超参数######
        super_wrong_width_num_pure = 16  # 纯过道最小宽度不满足的数量（切线法）
        super_wrong_width_num_all = 8  # 公共区域最小宽度不满足的数量（切线法）
        step = 300  # 模数

        super_area_ratio_100 = 0.39
        super_area_ratio_85_ = 0.2
        super_area_ratio_75 = 0.14
        super_area_ratio_50 = 0.1  # 过道面积/建筑面积，50分位数
        super_area_ratio_25 = 0.065  # 25分位数

        super_area_ratio_pub_all_100 = 0.67
        super_area_ratio_pub_all_75 = 0.47
        super_area_ratio_pub_all_75_ = 0.5

        super_vertice_ratio_100 = 0.35
        super_vertice_ratio_75 = 0.12
        super_vertice_ratio_50 = 0.086
        super_vertice_ratio_25 = 0.06

        path_pure, path_all, layout_area = self.get_path()
        # super_area_min, super_area_max, super_vertice_min, super_vertice_max = self.super_parameter_calculator(area=layout_area)

        # 可视化过道形态
        # for poly in [path_all]:
        #     x = [p[0] for p in list(poly.exterior.coords)]
        #     y = [p[1] for p in list(poly.exterior.coords)]
        #     plt.plot(x, y, color='black')
        #     plt.fill(x, y, color='lightgray', alpha=0.8)
        # plt.show()

        ###### 过道及公共空间评价————过道之间距离、最窄宽度、面积、复杂性 ######
        if path_all.geom_type == 'Polygon':
            iteration_path_all = [path_all]
        else:
            iteration_new = path_all.geoms
            iteration_path_all = [i for i in iteration_new]

        if path_pure.geom_type == 'Polygon':
            iteration_path_pure = [path_pure]
        else:
            iteration_pure_new = path_pure.geoms
            iteration_path_pure = [i for i in iteration_pure_new]

        # 纯过道复杂程度得分(顶点数量）
        vertice_list = []
        for sub_path in iteration_path_pure:
            vertices = sub_path.exterior.coords
            if vertices:
                vertice_num = len(list(vertices))-1  # shapely的坐标为了闭合，会多一个点
            else:
                vertice_num = 0
            vertice_list.append(vertice_num)
        vertice_num = sum(vertice_list)

        score_vertice_pure = self.curve_downward_2parm(
            x=vertice_num,
            x_min=super_vertice_ratio_25 * layout_area / 1000000,
            x_max=super_vertice_ratio_100 * layout_area / 1000000
        )

        # 纯过道最窄尺寸得分
        min_width_union = []  # 用来评价过道的宽度是否合适
        for t in iteration_path_pure:  # 找公共区域的最窄宽度
            if not t.is_empty:
                unreasonal_width_list = self.minimum_width(shape=t, boundary=self.boundary)  # 计算最窄宽度集合
                min_width_union = min_width_union + unreasonal_width_list
        wrong_width_num = len(min_width_union)  # 用基准尺寸减去当前尺寸得到还需要多少尺寸
        # print('width:', min_width_union, wrong_width_num)
        score_width_pure = self.curve_downward_2parm(x=wrong_width_num, x_min=0, x_max=super_wrong_width_num_pure)

        # 纯过道面积比例得分
        # print('path_area:', path_pure.area, super_area_ratio_25*layout_area, super_area_ratio_100*layout_area)
        score_area_pure = self.curve_downward_2parm(
            x=path_pure.area,
            x_min=super_area_ratio_25 * layout_area,
            x_max=super_area_ratio_85_ * layout_area
        )

        # print('平面总面积', layout_area)
        # print('纯过道面积', path_pure.area)
        # print('满分面积', super_area_ratio_25 * layout_area)
        # print('零分面积', super_area_ratio_100 * layout_area)
        # print('得分', score_area_pure)

        # 公共区域总面积比例得分
        score_area_all = self.curve_downward_2parm(
            x=path_all.area,
            x_min=super_area_ratio_pub_all_75_ * layout_area,
            x_max=super_area_ratio_pub_all_100 * layout_area
        )

        # self.visual_curve(func=self.curve_downward_2parm, lis_x=lis_x,
        #                   x_min=super_area_ratio_25 * layout_area,
        #                   x_max=super_area_ratio_100 * layout_area)

        # 公共区域最窄尺寸得分
        # min_width_union = []  # 用来评价过道的宽度是否合适
        # for t in iteration_path_all:  # 找公共区域的最窄宽度
        #     if not t.is_empty:
        #         unreasonal_width_list = self.minimum_width(shape=t, boundary=self.boundary)  # 计算最窄宽度集合
        #         min_width_union = min_width_union + unreasonal_width_list
        # wrong_width_num = len(min_width_union)  # 用基准尺寸减去当前尺寸得到还需要多少尺寸
        # # print('width:', min_width_union, wrong_width_num)
        # score_width_all = self.curve_downward_2parm(x=wrong_width_num, x_min=0, x_max=super_wrong_width_num_all)

        # 房间可达性评价
        distance_all = 0
        for r in self.rooms_valid:
            if not r.endswith('sub'):
                room = self.room_inner_storage.loc['poly', r]
                if (room.area > 0) and (path_all.area > 0):
                    dis = room.distance(path_all)
                    distance_all += dis
        score_room_accessibility = (300 / (300 + distance_all))

        # 公共区域是否联通（连续变量）
        # iteration_path_pure_add = iteration_path_pure.copy()  # 纯粹过道
        iteration_path_pure_add = iteration_path_all.copy()  # 包含公共区域的过道
        for p in self.entrances:
            iteration_path_pure_add.append(self.df_shape.loc['rec', p])  # 包含所有入口的过道

        path_pure_entrance = ops.unary_union(iteration_path_pure_add)
        if path_pure_entrance.geom_type == 'Polygon':
            score_path_pure_blocked = 1
        else:
            iteration_pure_entrance_new = path_pure_entrance.geoms
            iteration_path_pure_entrance = [i for i in iteration_pure_entrance_new]

            distance_sum = 0
            for idx, path_block in enumerate(iteration_path_pure_entrance):
                distance_sub = []
                for path_block_ in iteration_path_pure_entrance[idx+1:]:
                    if (path_block.area > 1e-10) and (path_block_.area > 1e-10):
                        dis = path_block.distance(path_block_)  # 求两个过道体块距离
                        if dis == 0:  # 排除掉虽然相交，但可通过尺寸不够的情况
                            length_intersection = path_block.exterior.intersection(path_block_.exterior).length
                            if length_intersection < 600:  # 如果有相交长度
                                distance_sub.append(600 - length_intersection)
                        else:
                            distance_sub.append(dis)
                if len(distance_sub) > 0:
                    distance_sub_min = min(distance_sub)
                    distance_sum += distance_sub_min
            score_path_pure_blocked = 300 / (distance_sum + 300)

        # 其它指标
        lis = [
            score_vertice_pure,
            score_width_pure,
            score_area_pure * 2.5,
            score_area_all,
        ]
        lis_index = [
            'vertices_pure',
            'width_pure',
            'area_pure',
            'area_all',
        ]
        series_path = pd.Series(np.array(lis), index=lis_index)

        # 可达性指标
        lis_access = [
            score_path_pure_blocked,
            score_room_accessibility
        ]
        lis_index_access = [
            'path_pure_blocked',
            'path_room_access'
        ]
        series_path_access = pd.Series(np.array(lis_access), index=lis_index_access)
        # print(series_path)
        return series_path, series_path_access

    @ normalization_living_dining_shape
    def living_dining_unit_score(self):
        # 客厅、餐厅联合体评价, 垂直射线法，计算共享射线占比
        if ('living' in self.df_shape.columns) and ('dining' in self.df_shape.columns):
            shape_others_union = unary_union(self.room_inner_storage)
            shape_living = self.df_shape.loc['poly', 'living']
            shape_dining = self.df_shape.loc['poly', 'dining']
            shape_union = unary_union([shape_dining, shape_living])

            hori_lines, vert_lines = self.creat_divide_lines(boundary=shape_union)
            x_min, y_min, x_max, y_max = shape_dining.bounds
            dining_vert_size = y_max - y_min
            dining_hor_size = x_max - x_min

            def score_calculator(line_set, size_din):
                if line_set.geom_type == 'LineString':
                    iteration = [line_set]
                else:
                    iteration = line_set.geoms
                line_container = []  # 通过线的相切情况，判断相交比例
                for line in iteration:
                    intersect_living = line.difference(shape_living)
                    if intersect_living.length < line.length:
                        intersect_living_dining = intersect_living.difference(shape_dining)
                        if intersect_living_dining.length < intersect_living.length:
                            intersect_others = line.difference(shape_others_union)
                            if not intersect_others.geom_type == 'MultiLineString':
                                line_container.append(line)

                score = (len(line_container) * 300) / (size_din + 1e-6)

                # debug用，画出客厅餐厅和切割线
                # for poly in [shape_dining, shape_living]:
                #     x = [p[0] for p in list(poly.exterior.coords)]
                #     y = [p[1] for p in list(poly.exterior.coords)]
                #     plt.plot(x, y, color='black')
                #     plt.fill(x, y, color='lightgray', alpha=0.8)
                #
                # for sub in [line_container]:
                #     for line in sub:
                #         x = [p[0] for p in list(line.coords)]
                #         y = [p[1] for p in list(line.coords)]
                #         plt.plot(x, y, color='black')
                #         plt.fill(x, y, color='lightgray', alpha=0.8)
                # plt.show()
                return score
            score_vert = score_calculator(line_set=hori_lines, size_din=dining_vert_size)
            score_hor = score_calculator(line_set=vert_lines, size_din=dining_hor_size)
            score_living_dining = max([score_vert, score_hor])
        else:
            score_living_dining = 0

        final_score = self.curve_upward_2parm(x=score_living_dining, x_min=0, x_max=0.7)
        return final_score

    # 下弯余弦曲线
    def curve_upward_2parm(self, x, x_min, x_max):
        if x <= x_min:
            score = 0
        elif x >= x_max:
            score = 1
        else:
            x_std = (np.pi / 2) * ((x - x_min) / (x_max - x_min))
            score = np.sin(x_std)
        return score

    def minimum_width(self, shape, boundary):
        lines_horizontal, lines_vertical = self.creat_divide_lines(boundary=boundary)
        # 等分线同边界轮廓取交集
        lines_horizontal_cut = shape.intersection(lines_horizontal)
        lines_vertical_cut = shape.intersection(lines_vertical)

        shape_coord = shape.exterior.coords
        shape_lines = geometry.LineString(np.array(shape_coord))

        lines_horizontal_final = lines_horizontal_cut.difference(shape_lines)
        lines_vertical_final = lines_vertical_cut.difference(shape_lines)

        # 计算最窄部分长度
        length_list = []
        for line_cut in [lines_horizontal_final, lines_vertical_final]:
            if line_cut.geom_type == 'LineString':
                iteration = [line_cut]
            else:
                iteration = line_cut.geoms

            for i, item in enumerate(iteration):
                if item.length > 0:  # 排除 line_cut 为空值情况
                    length_list.append(item.length)
                else:
                    pass
        if len(length_list) == 0: # 排除列表为空集的情况
            return []
        length_list_less900 = [item for item in length_list if item < 900 and item > 0]

        return length_list_less900

    def creat_divide_lines(self, boundary, if_horizontal=True, if_vertical=True):
        '''创建一个矩形内部水平和垂直等间距的等分线'''
        # 得到边界轮廓的长、宽
        width = boundary.bounds[2]
        depth = boundary.bounds[3]
        # 得到边长等分坐标列表
        array_x = np.arange(start=150, stop=width, step=300)
        array_y = np.arange(start=150, stop=depth, step=300)

        lines_vertical = None
        lines_horizontal = None
        # 得到水平和垂直等分线集合
        if if_vertical:
            coord_horizontal_bottom = np.array([array_x, np.zeros(len(array_x))]).T
            coord_horizontal_top = np.array([array_x, np.ones(len(array_x)) * depth]).T
            points_unit_horizontal = np.array([coord_horizontal_bottom, coord_horizontal_top]).transpose((1, 0, 2))
            lines_vertical = geometry.MultiLineString(points_unit_horizontal.tolist())

        if if_horizontal:
            coord_vertical_left = np.array([np.zeros(len(array_y)), array_y]).T
            coord_vertical_right = np.array([np.ones(len(array_y)) * width, array_y]).T
            points_unit_vertical = np.array([coord_vertical_left, coord_vertical_right]).transpose((1, 0, 2))
            lines_horizontal = geometry.MultiLineString(points_unit_vertical.tolist())

        # 画出path和切割线
        # print(boundary)
        # x = [p[0] for p in list(boundary.exterior.coords)]
        # y = [p[1] for p in list(boundary.exterior.coords)]
        # plt.plot(x, y, color='black')
        # plt.fill(x, y, color='lightgray', alpha=0.8)
        #
        # x = [p[0] for p in list(shape.exterior.coords)]
        # y = [p[1] for p in list(shape.exterior.coords)]
        # plt.plot(x, y, color='black')
        # plt.fill(x, y, color='lightgray', alpha=0.8)
        #
        # for i in points_unit_vertical:
        #     x = [j[0] for j in i]
        #     y = [j[1] for j in i]
        #     plt.plot(x, y)
        # plt.show()

        return lines_horizontal, lines_vertical

    # 获取过道
    def get_path(self):
        without_boundary = self.df_shape.drop(columns=['boundary']).copy()
        pub_rooms = list(set(self.df_shape.columns.tolist()).intersection(set(self.rooms_pub)))  # 筛选出存在的可做过道的房间

        without_living_dining = without_boundary.drop(columns=pub_rooms).copy()
        shapes_union = unary_union(without_boundary.loc['rec'].values)
        path_pure = self.boundary.difference(shapes_union)
        path_all = self.boundary.difference(unary_union(without_living_dining.values))
        self.public = path_all
        self.passway = path_pure
        layout_area = unary_union(self.room_inner_storage.loc['rec']).area + path_pure.area
        return path_pure, path_all, layout_area

    # 下弯直线
    def curve_stright_line_downwards(self, x, x_min, x_max):
        if x < x_min:
            score = 1
        elif (x >= x_min) and (x < x_max):
            score = ((x_max - x) / (x_max - x_min))
        else:
            score = 0
        return score

    # 下弯余弦曲线
    def curve_downward_2parm(self, x, x_min, x_max):
        if x <= x_min:
            score = 1
        elif x >= x_max:
            score = 0
        else:
            x_std = (np.pi / 2) * ((x - x_min) / (x_max - x_min))
            score = np.cos(x_std)
        return score

    # 获得本户型的最大过道面积，以及最大定点数的超参数
    def super_parameter_calculator(self, area):
        '''
        :param area: 户型面积
        :return:
        '''
        max_area = 200000000  # 最大户型面积
        min_area = 20000000  # 最小户型面积

        min_path_area = 4000000   # 最小户型过道面积均值
        max_path_area = 15000000  # 最大户型过道面积均值
        interval_area = 3000000

        min_path_vertices = 4     # 最少过道顶点数
        max_path_vertices = 12    # 最多过道顶点数
        interval_vertices = 3

        if area > max_area:
            area = max_area
        if area < min_area:
            area = min_area

        super_area = (max_path_area - min_path_area) * ((area - min_area) / (max_area - min_area)) + min_path_area
        super_area_min = super_area - interval_area
        super_area_max = super_area + interval_area
        super_vertices = (max_path_vertices - min_path_vertices) * ((area - min_area) / (max_area - min_area)) + min_path_vertices
        super_vertices_min = super_vertices - interval_vertices
        super_vertices_max = super_vertices + interval_vertices
        # print(area, super_area_min, super_area_max, super_vertices_min, super_vertices_max, '----------')
        return super_area_min, super_area_max, super_vertices_min, super_vertices_max

    def visual_curve(self, func, lis_x, x_min, x_max):
        lis_y = []
        for i in lis_x:
            y = func(i, x_min, x_max)
            lis_y.append(y)
        plt.plot(lis_x, lis_y)
        plt.show()
        time.sleep(1)


if __name__ == '__main__':
    import os
    from tools_layout_modeling import layout2geopandas, adjacent_matrix_shapely

    # path = '../cases_for_test_large/2_improved/'
    # path = 'D:\\ONGOING\\RL_house\\Supervised_cnn_city_rural_house\\data_test\\'
    path = 'C:/Users/SHU/Desktop/2025_05_20_11_51_38/results/city/'

    for root, dirs, files in os.walk(path):
        for file in files:
            if file == 'city_large_上海_万科_上海万科新城_1梯2户_6层-230.89平-237-1-1.xlsx':
            # if file.endswith('xlsx') and file != 'rural_floor2_E-I-95.xlsx':

                df = pd.read_excel(path + file, index_col=0, sheet_name='floor1')

                geo_layout = layout2geopandas(layout_info=df)
                case = PathShapeScore(df_shape=geo_layout)
                out = case.path_shape_total_score()
                print(file)
                print(out)
                living_dining = case.living_dining_unit_score()
                # print(living_dining)
                print('----------')






