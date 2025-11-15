"""
主要目的: 提取出可放置房间的节点信息
get_leaf_node类：
    属性：
        户型坐标信息layout_info (dataframe)
        户型房间矢量模型layout_model (dataframe)

传入：准备放置的新房间名称（str）、当前已经放置完毕的户型信息(坐标、尺寸)
返回：可以放置房间的点集合（dataframe）
"""

import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import shapely
from shapely import ops
from shapely import geometry

from tools_layout_modeling import layout2geopandas
from tools_layout_modeling import build_room, layout_change
from pruning.get_leaf_node_parameters import room_relation, room_sizes, area_limits, window_limits
# from get_leaf_node_parameters import room_relation, room_sizes, area_limits, window_limits

pd.set_option("expand_frame_repr", False)


# @timethis
class get_available_layout:
    def __init__(self, new_room_name, env_info, room_info=None, control=None, layout_down=None):
        """
        Args:
            new_room_name: 待生成的房间
            env_info: 设计条件（环境）
            room_info: 已生成的房间信息
            control: 偏好控制参数
        """
        self.new_room = new_room_name  # 房间名称

        if "gate" in env_info.columns:
            self.env_info = env_info[env_info.columns.drop("gate")]  # 去除entrance信息
        else:
            self.env_info = env_info

        self.room_info = room_info
        self.layout_down = layout_down
        self.control = control  # 邻接关系控制
        self.room_adjacent_matrix = room_relation()

        # 拼接环境和房间
        if (room_info is not None) and (not room_info.empty):
            self.layout_info = pd.concat([self.env_info, self.room_info], axis=1)
        else:
            self.layout_info = self.env_info
        self.poly_layout = layout2geopandas(layout_info=self.layout_info)

        # 得到用来适配graph剪枝的poly_layout信息，包含了四个boundary
        layout_info_boundary_plus = self.add_direction_cubes(layout=copy.deepcopy(self.layout_info))
        self.poly_layout_boundary_plus = layout2geopandas(layout_info=layout_info_boundary_plus)

        self.leaf_points = None

        # 得到包含全部房间信息的shapley格式dataframe
        for i, j in enumerate(self.layout_info.columns):
            x = self.layout_info.loc["x", j]
            y = self.layout_info.loc["y", j]
            w = self.layout_info.loc["w", j]
            d = self.layout_info.loc["d", j]

            room = build_room(x, y, w, d)
            self.poly_layout.iloc[0, i] = room

        # 初始环境的空白区域
        self.poly_boundary = self.poly_layout.loc["rec", "boundary"]
        env_other_room_names = self.env_info.columns.drop("boundary")
        poly_other_env = self.poly_layout.loc["rec", env_other_room_names]  # 除了boundary外的所有环境要素集合体
        self.poly_other_env_unit = ops.unary_union(poly_other_env)
        self.poly_blank_env = self.poly_boundary.difference(self.poly_other_env_unit)  # 初始环境轮廓

        # 去除white_m的空白区
        lis_poly_white_m = [self.poly_layout.loc['rec', i] for i in self.poly_layout.columns if i.startswith('white')]
        white_m_union = ops.unary_union(lis_poly_white_m)
        self.poly_blank_env_without_white_m = self.poly_blank_env.difference(white_m_union)  # 不包含white_m的初始环境轮廓
        self.poly_pub = self.get_path()

        # 一些房间名称，用于筛选可放置点
        self.names_neutral = ["courtyard"]  # 中立房间，指介于设计条件和生成对象之间的参数
        self.names_entrance = ["entrance", "entrance_sub", "porch"]  # 入口，可以放在边界外（永远四个象限）
        self.names_sub = ["bath1_sub"]  # 从属房间，只能放在主房间内
        self.names_point_pruning = [
            "white_m1", "white_m2", "white_m3", "white_m4", "bath1_sub", "garage",
            "dining", "storeroom", "study_room",
        ]  # 定义角点剪枝的房间，非边剪枝

        # 将所有房间合并为一个整体（用于合规性判定等）
        if self.room_info is not None:
            poly_rooms_rest = self.poly_layout.loc["rec", self.room_info.columns.tolist()]
            self.rooms_deployed_union = shapely.ops.unary_union(poly_rooms_rest)

            self.rooms_env_deployed_union = ops.unary_union([self.rooms_deployed_union, self.poly_other_env_unit])
        else:
            self.rooms_env_deployed_union = self.poly_other_env_unit

        # 提取出所有白体
        self.light_names = []
        for c in self.poly_layout.columns:
            if (c.startswith("white")) or (c == "courtyard") or (c == "porch"):
                self.light_names.append(c)

        # 判断是否是二层，如果是的话，需要排除掉不符合墙体约束的点
        if self.layout_down is not None:
            dic_down = self.get_down_stair_points()
            if dic_down is not None:
                self.lis_wall_points_down = dic_down['points_wall']
                self.lis_corner_points_down = dic_down['points_corner']
            else:
                self.lis_wall_points_down = []
                self.lis_corner_points_down = []

        # 得到所有实体房间的交集（计算是否遮挡入口）
        lis_exclude = ["living", "dining", "hallway"]
        self.lis_room_entity = []
        for r in self.room_info.columns:
            if (r not in lis_exclude) and (not r.startswith('white')):
                self.lis_room_entity.append(self.poly_layout.loc['rec', r])

    def add_direction_cubes(self, layout):
        """将layout_info改成适配graph的数据格式"""
        x, y, w, d = layout.loc[:, 'boundary']
        cube_west = [x-1200, y, 1200, d]
        cube_east = [x+w, y, 1200, d]
        cube_south = [x, y-1200, w, 1200]
        cube_north = [x, y+d, w, 1200]
        layout = layout.drop('boundary', axis=1)  # 减去boundary
        add_columns = ['boundary_west', 'boundary_east', 'boundary_south', 'boundary_north',]
        layout[add_columns] = np.array([cube_west, cube_east, cube_south, cube_north]).T
        # print(self.layout_ori)
        return layout

    def get_down_stair_points(self):
        """
        Args:
            df_info_now: 输入二层房间信息
        Returns:
        """
        # 所有实体房间名称
        names_room_down_stair = [
            'room1', 'room2', 'room3', 'room4', 'study_room', 'garage', 'kitchen',
            'staircase', 'bath1', 'bath2', 'bath1_sub', 'storeroom', 'hallway'
        ]

        df_env_poly = self.poly_layout
        poly_boundary = df_env_poly.loc['rec', 'boundary']  # boundary范围
        df_poly_others = copy.deepcopy(df_env_poly).drop('boundary', axis=1)
        poly_env_union = ops.unary_union(df_poly_others.loc['rec', :])
        poly_interior = poly_boundary.difference(poly_env_union)  # 得到二层室内空间
        # print(self.layout_info)
        width = self.layout_info.loc['w', 'boundary'].item() / 300
        depth = self.layout_info.loc['d', 'boundary'].item() / 300
        x_ori = self.layout_info.loc['x', 'boundary'].item()
        y_ori = self.layout_info.loc['y', 'boundary'].item()

        # 使用列表推导式生成坐标矩阵,得到矩阵中每个点的坐标列表
        matrix = [[(x_ori + x * 300, y_ori + y * 300) for x in range(int(width + 1))] for y in range(int(depth + 1))]
        matrix.reverse()

        # 将点阵合并成一个列表
        lis_coord = []
        for p in matrix:
            lis_coord = lis_coord + p
        multi_points = ops.MultiPoint(lis_coord)
        multi_points_interior = multi_points.intersection(poly_interior)  # 得到纯室内的点阵

        # 查找每个房间拐角点
        room_names = [i for i in names_room_down_stair if i in self.layout_down]
        lis_points_corner = []  # 所有实体房间拐角点
        for r in room_names:
            x, y, w, d = self.layout_down.loc[:, r]
            p1 = [x, y]
            p2 = [x, y + d]
            p3 = [x + w, y]
            p4 = [x + w, y + d]
            for p in [p1, p2, p3, p4]:
                point = geometry.Point(p)
                if point.intersects(multi_points_interior) and (p not in lis_points_corner):
                    lis_points_corner.append(p)

        if poly_interior.geom_type == 'MultiPolygon':
            iteration = poly_interior.geoms
        else:
            iteration = [poly_interior]

        # 查找边界拐角点
        for poly in iteration:
            for po in ops.MultiPoint(poly.exterior.coords).geoms:
                if po.intersects(multi_points_interior) and (po not in lis_points_corner):
                    lis_points_corner.append([po.x, po.y])

        # 得到所有边界上的点坐标, 包括边界轮廓线，以及所有实体房间的边界
        outlines = [line.exterior for line in iteration]  # 自动包含了边界轮廓线
        df_poly_rooms = layout2geopandas(layout_info=self.layout_down[room_names])
        for room in df_poly_rooms.columns:
            poly_room = df_poly_rooms.loc['rec', room]
            room_outline = poly_room.exterior
            outlines.append(room_outline)

        union_outline = ops.unary_union(outlines)
        geo_points_online = multi_points_interior.intersection(union_outline)

        if (not geo_points_online.is_empty) and (not multi_points_interior.is_empty):
            lis_points_online = [[i.x, i.y] for i in geo_points_online.geoms]  # 所有在边界线上的点
            lis_points_interior_all = [[i.x, i.y] for i in multi_points_interior.geoms]  # 所有在室内的点

            dic = {
                'points_corner': lis_points_corner,  # 所有拐点
                'points_wall': lis_points_online,  # 所有墙线上的点
                'points_interior_all': lis_points_interior_all,  # 室内的全部点
                'poly_boundary': poly_boundary,  # boundary的polygon
                'poly_interior': poly_interior  # 室内区域的polygon
            }
        else:
            return None
        return dic

    def get_shapely_single_polygon(self, geom):
        if geom.geom_type == 'Polygon':
            return [geom]
        elif geom.geom_type == 'MultiPolygon':
            return [poly for poly in geom.geoms]

    def get_points_on_corner(self, poly_blank):
        if poly_blank.geom_type == "Polygon":
            iteration = [poly_blank]
        else:
            iteration = poly_blank.geoms

        lis_points = []
        for poly in iteration:
            points = list(poly.exterior.coords)
            lis_points_sub = [[i[0], i[1]] for i in points]
            lis_points = lis_points + lis_points_sub
        return lis_points

    def get_points_on_edge(self, poly_boundary, poly_blank):
        """得到环境的边界轮廓上的点坐标，以及个房间角点坐标"""
        width = self.layout_info.loc["w", "boundary"].item() / 300
        depth = self.layout_info.loc["d", "boundary"].item() / 300
        x_ori = self.layout_info.loc["x", "boundary"].item()
        y_ori = self.layout_info.loc["y", "boundary"].item()

        # 使用列表推导式生成坐标矩阵,得到矩阵中每个点的坐标列表
        matrix = [[(x_ori + x * 300, y_ori + y * 300) for x in range(int(width + 1))] for y in range(int(depth + 1))]
        matrix.reverse()

        # 将点阵合并成一个列表
        lis_coord = []
        for p in matrix:
            lis_coord = lis_coord + p
        multi_points = geometry.MultiPoint(lis_coord)
        # 将点阵（multipoint）与边界外轮廓相交，得到在边界轮廓上的点的列表

        # 得到边界上的所有点
        lis_points_on_edge = []
        if poly_boundary.geom_type == "MultiPolygon":
            iteration_boundary = poly_boundary.geoms
        else:
            iteration_boundary = [poly_boundary]
        for b in iteration_boundary:
            outline_boundary_points = b.exterior.coords[:]
            outline_boundary = geometry.LinearRing(outline_boundary_points)  # 边界上的线
            outline_boundary_sub = outline_boundary.intersection(poly_blank)
            points_on_edge = multi_points.intersection(outline_boundary_sub)  # 边界上的点坐标
            lis_points_on_edge.append(points_on_edge)

        points_on_edge_all = ops.unary_union(lis_points_on_edge)

        # 得到符合要求的所有点
        lis_points = []
        if points_on_edge_all.geom_type == "MultiPoint":
            point_iter = points_on_edge_all.geoms
        elif points_on_edge_all.is_empty:
            point_iter = []
        else:
            point_iter = [points_on_edge_all]

        for p in point_iter:
            lis_points.append([p.x, p.y])

        # 得到剩余的未被填充的空白区域上的角点
        if poly_blank.geom_type == "Polygon":
            iteration = [poly_blank]
        else:
            iteration = poly_blank.geoms

        for poly in iteration:
            coord = geometry.MultiPoint(poly.exterior.coords[:])
            points = coord.intersection(multi_points)
            if points.geom_type == "MultiPoint":
                iter_point = points.geoms
            else:
                iter_point = [points]
            for po in iter_point:
                if [po.x, po.y] not in lis_points:
                    lis_points.append([po.x, po.y])
        return lis_points

    # @timer_decorator
    def get_deploy_points(self):
        """放置规则为，可放置的点，既在边界轮廓上，又为空白区域角点，又满足邻接关系强约束，且符合采光要求"""
        # 得到待放置房间的空白区域
        boundary = self.poly_layout.loc["rec", "boundary"]
        if (self.new_room not in self.names_entrance) and (self.new_room not in self.names_sub):
            rest_columns = self.poly_layout.columns.drop("boundary")
            poly_rooms_rest = self.poly_layout.loc["rec", rest_columns]
            rooms_rest_union = shapely.ops.unary_union(poly_rooms_rest)
            poly_blank = boundary.difference(rooms_rest_union)

            if poly_blank.is_empty:  # 排除掉整个边界已填满，无空白空间剩余情况
                return None
            # 判断可放置房间的点位及象限
            poly_env_all = self.get_boundary_all_whites()

            # 判断用哪种剪枝方式
            if not self.new_room in self.names_point_pruning:
                points_blank = self.get_points_on_edge(poly_boundary=poly_env_all, poly_blank=poly_blank)
            else:
                points_blank = self.get_points_on_corner(poly_blank=poly_blank)

        else:
            poly_blank = self.get_boundary_all_whites()

            if poly_blank.is_empty:  # 排除掉整个边界已填满，无空白空间剩余情况
                return None

            # 如果是入口，可放置的区域包括了当前的环境边界（包括white_m）上的所有的点，以及所有象限
            if not self.new_room in self.names_point_pruning:  # 判断用哪种剪枝方式
                points_blank = self.get_points_on_edge(poly_boundary=poly_blank, poly_blank=poly_blank)
            else:
                points_blank = self.get_points_on_corner(poly_blank=poly_blank)

        # 判断是否是二层，如果是的话，需要排除掉不符合墙体约束的点
        if self.layout_down is not None:
            points_blank_screen = []
            for point_up in points_blank:
                if point_up in self.lis_corner_points_down:
                    points_blank_screen.append(point_up)
            points_blank = points_blank_screen

        # if self.new_room == "entrance":
        #     px = []
        #     py = []
        #     for p in points_blank:
        #         px.append(p[0])
        #         py.append(p[1])
        #     plt.scatter(px, py, s=40, c="r")
        #     plt.show()

        final_leaf_points = np.empty((0, 2))
        # 根据邻接矩阵筛选放置点，-1代表任意一个，1代表必须贴边(不包含white_m)，2代表必须贴边（包含white_m), 3代表附属用房，0 代表无所谓
        if self.new_room not in self.room_adjacent_matrix.T.columns:
            raise ValueError(f"room name is wrong ! {self.new_room}")
        relative_rooms = self.room_adjacent_matrix.loc[self.new_room, :]

        # 判断，并构建房间弱约束列表
        lis_names_available = relative_rooms[relative_rooms == -1].index.tolist()
        if not lis_names_available:
            lis_names_available = None

        # 判断，并筛选边界约束点
        adj_sub = self.room_adjacent_matrix.loc[self.new_room, :]

        # 根据约束条件筛选可放置房间的所有点
        poly_boundary_outline = None
        if self.room_adjacent_matrix.loc[self.new_room, "boundary"] == 0:  # 代表无boundary约束
            # 判断并筛选boundary外的其它强约束点
            poly_boundary_outline = None

        elif self.room_adjacent_matrix.loc[self.new_room, "boundary"] == 1:  # 代表不包含white_m的boundary约束
            # 筛选出边界轮廓
            rooms_excluded = list(self.env_info.columns.drop("boundary"))  # 筛选出边界轮廓
            rooms_excluded_exist = list(
                set(rooms_excluded).intersection(set(self.layout_info.columns)))  # 选取存在于户型中的点
            poly_rooms_excluded = self.poly_layout.loc["rec", rooms_excluded_exist]
            poly_rooms_excluded_union = ops.unary_union(poly_rooms_excluded)
            poly_boundary_outline = boundary.difference(poly_rooms_excluded_union)

        elif self.room_adjacent_matrix.loc[self.new_room, "boundary"] == 2:  # 代表包含white_m的boundary约束
            # 筛选出除去white_m的边界轮廓
            poly_boundary_outline = self.get_boundary_all_whites()

        # 判断是否为嵌套房间，如独立卫生间
        if not 3 in adj_sub.values.tolist():  # 代表该房间不为从属性房间
            leaf_points_raw = self.points_screen(room_constrain_weak=lis_names_available,
                                                 boundary_constrain=poly_boundary_outline,
                                                 points_on_blank=points_blank)

            leaf_points = self.points_screen_graph(points_available=leaf_points_raw, graph_constrain=self.control)

            # 确定可以放置的象限, 并建立最终的leaf_points集合
            for p, q in enumerate(leaf_points):
                out = self.point_quadrant_coline(point=q, poly_scope=poly_blank)
                final_leaf_points = np.concatenate([final_leaf_points, out], axis=0)

        else:  # 独立卫生间
            """
            1.先区分是否贴着环境轮廓。与至少两个主要白体相邻，或与不采光边相邻的情况被判定为“贴着环境轮廓”
            2.如贴着环境轮廓，则放置顺序依次为：不采光点>white_third/fourth/white_m>white_north>white_south
            3.如不贴着环境轮廓，则放置顺序以此为：所有不与主白体相邻点
            """
            strong_constrain_name = adj_sub[adj_sub == 3].index.tolist()  # Room_Rlation 文件 每行只能有一个 1或2
            if not strong_constrain_name:  # 确定独卫的主房间存在
                print("get_leaf_node: The depended room of bath_sub is not exist")
                return None

            # 得到原始边界轮廓
            if self.poly_blank_env.geom_type == 'MultiPolygon':
                lis_poly_blank_env = self.poly_blank_env.geoms
            else:
                lis_poly_blank_env = [self.poly_blank_env]

            room_main = self.poly_layout.loc["rec", strong_constrain_name[0]]  # sub房间所从属的主房间
            room_main_corners = ops.MultiPoint(room_main.exterior.coords[:])  # 得到房间角点

            names_white = [
                'white_fourth', 'white_third', 'white_west', 'white_east',
                'white_m1', 'white_m2', 'white_north', 'white_south'
            ]  # 这个排序很重要，表示与主卫相邻的优先级，不要乱动

            poly_all_whites = [w for w in names_white if w in self.poly_layout.columns]
            poly_white = self.poly_layout.loc["rec", poly_all_whites]
            poly_white_union = ops.unary_union(poly_white)

            # 判断主卧是否贴边（即有三个边不与边界相邻）
            judge_stick_boundary = True
            # 计算主卧与白体相邻的数量
            counter_white = 0
            line_room_main = room_main.exterior
            lis_line_room = list(line_room_main.coords)
            lis_lines = [ops.LineString([i, lis_line_room[idx + 1]]) for idx, i in enumerate(lis_line_room[:-1])]  # 得到主卧的四个边线段

            for line in lis_lines:
                if line.intersection(poly_white_union).length > 300:
                    counter_white += 1
            line_room_main_cut = line_room_main.difference(poly_white_union)  # 主卧的外轮廓减去与所有白体相邻部分

            for poly_blank_env_sub in lis_poly_blank_env:
                line_blank_env = poly_blank_env_sub.exterior
                if (line_room_main_cut.intersection(line_blank_env).length < 300) and (counter_white < 2):  # 说明主卧不贴边
                    judge_stick_boundary = False

            points_recorder = []  # 第一轮筛选
            for point in room_main_corners.geoms:
                # 判断主卧是否贴边，贴边与否的应对策略不同
                if judge_stick_boundary is False:  # 说明主卧不贴边，则只在内侧点放置主卫
                    points_recorder.append(point)

                else:  # 说明主卧贴边，则只允许放在边界上
                    env_polys = self.get_shapely_single_polygon(geom=self.poly_blank_env_without_white_m)
                    for env_poly in env_polys:
                        if point.intersects(env_poly.exterior):
                            points_recorder.append(point)

            points_recorder1 = []  # 第二轮筛选
            for p1 in points_recorder:  # 如果该点不与任何白体相交，则加入该点（说明靠在采光面内测）
                if not p1.intersects(poly_white_union):
                    points_recorder1.append([p1.x, p1.y])

            if len(points_recorder1) == 0:  # 如果没筛选出不与白体相交的点，则放置所有点
                for pw in poly_white:
                    for p2 in points_recorder:
                        if p2.intersects(pw):
                            points_recorder1.append([p2.x, p2.y])
                    if len(points_recorder1) >= 1:
                        break

            points_recorder2 = self.points_screen_graph(points_available=points_recorder1, graph_constrain=self.control)


            # 确定可以放置的象限, 并建立最终的leaf_points集合
            for p, q in enumerate(points_recorder2):
                area_new = ops.unary_union([poly_blank, room_main])  # 空白区域加上强约束房间
                if self.new_room == "bath1_sub":
                    out = self.point_quadrant_subroom(point=q, area=area_new, constrain_room=room_main,
                                                      if_boundary=True)
                else:
                    raise Exception("目前只支持主卧独卫")
                final_leaf_points = np.concatenate([final_leaf_points, out], axis=0)

        # 排除 leaf_points为[]的情况
        if final_leaf_points.size == 0:
            return None

        # 转为dataframe
        columns_name = ["p" + str(m) for m in range(len(final_leaf_points))]
        df_final_leaf_points = pd.DataFrame(final_leaf_points.T, columns=columns_name, index=["point", "quadrant"])
        return df_final_leaf_points

    def get_boundary_all_whites(self):
        """ 筛选出剪掉white_m的边界轮廓 """
        boundary = self.poly_layout.loc["rec", "boundary"]
        rooms_excluded = list(self.env_info.columns.drop("boundary"))  # 筛选出边界轮廓
        if self.room_info is not None:
            for w in self.room_info.columns:
                if w.startswith("white"):
                    rooms_excluded.append(w)
        rooms_excluded_exist = list(
            set(rooms_excluded).intersection(set(self.layout_info.columns)))  # 选取存在于户型中的点
        poly_rooms_excluded = self.poly_layout.loc["rec", rooms_excluded_exist]
        poly_rooms_excluded_union = ops.unary_union(poly_rooms_excluded)
        poly_boundary_outline = boundary.difference(poly_rooms_excluded_union)
        return poly_boundary_outline

    def point_quadrant(self, point, area):
        """房间放置点位及象限判定，后一个房间可以不与已有边界共线"""
        point_quadrant1 = [point[0] + 1, point[1] + 1]
        point_quadrant2 = [point[0] - 1, point[1] + 1]
        point_quadrant3 = [point[0] - 1, point[1] - 1]
        point_quadrant4 = [point[0] + 1, point[1] - 1]

        lis = [point_quadrant1, point_quadrant2, point_quadrant3, point_quadrant4]
        new_points = []
        for i, j in enumerate(lis):
            point_test = geometry.Point(j[0], j[1])
            if point_test.within(area):
                new_points.append([point, i + 1])
        array = np.array(new_points, dtype=object)
        return array

    def point_quadrant_coline(self, point, poly_scope):
        """房间放置点位及象限判定，后一个房间必须与已有边界共线，排除悬空房间"""
        small_rectangle_q1 = build_room(cord_x=point[0], cord_y=point[1], len_x=1, len_y=1, quadrant=1)
        small_rectangle_q2 = build_room(cord_x=point[0], cord_y=point[1], len_x=1, len_y=1, quadrant=2)
        small_rectangle_q3 = build_room(cord_x=point[0], cord_y=point[1], len_x=1, len_y=1, quadrant=3)
        small_rectangle_q4 = build_room(cord_x=point[0], cord_y=point[1], len_x=1, len_y=1, quadrant=4)

        lis = [small_rectangle_q1, small_rectangle_q2, small_rectangle_q3, small_rectangle_q4]
        if poly_scope.geom_type == "Polygon":
            iteration = [poly_scope]
        else:
            iteration = poly_scope.geoms
            iteration = [i for i in iteration]

        new_points = []
        # 判断该房间是否为中立房间或入口（中立房间的象限判定逻辑不同）
        if (self.new_room not in self.names_neutral) and (self.new_room not in self.names_entrance):
            for i, small_rec in enumerate(lis):
                for j in iteration:
                    if (small_rec.intersection(j).area > 0) and (small_rec.intersection(j.exterior).length > 0):
                        new_points.append([point, i + 1])
        elif self.new_room in self.names_neutral:
            for i, small_rec in enumerate(lis):
                if small_rec.intersection(self.poly_blank_env).area > 0:
                    new_points.append([point, i + 1])
        elif (self.new_room in self.names_entrance) and self.new_room != "porch":  # 入口房间只放置在poly_blank外侧
            for i, small_rec in enumerate(lis):
                if not (small_rec.intersection(poly_scope).area > 0):
                    new_points.append([point, i + 1])
        elif self.new_room == "porch":
            for i, small_rec in enumerate(lis):
                new_points.append([point, i + 1])

        array = np.array(new_points, dtype=object)
        return array

    def point_quadrant_subroom(self, point, area, constrain_room, if_boundary=False):
        """附属房间的可选点位及象限判定"""
        small_rectangle_q1 = build_room(cord_x=point[0], cord_y=point[1], len_x=1, len_y=1, quadrant=1)
        small_rectangle_q2 = build_room(cord_x=point[0], cord_y=point[1], len_x=1, len_y=1, quadrant=2)
        small_rectangle_q3 = build_room(cord_x=point[0], cord_y=point[1], len_x=1, len_y=1, quadrant=3)
        small_rectangle_q4 = build_room(cord_x=point[0], cord_y=point[1], len_x=1, len_y=1, quadrant=4)
        constrain_room_outline = constrain_room.exterior

        lis = [small_rectangle_q1, small_rectangle_q2, small_rectangle_q3, small_rectangle_q4]
        new_points = []
        for i, small_rec in enumerate(lis):
            if small_rec.within(area):
                if small_rec.intersection(constrain_room_outline).length > 0:
                    new_points.append([point, i + 1])
        if not new_points:  # 考虑没有筛选出放置点的情况
            return np.empty((0, 2))
        array = np.array(new_points, dtype=object)
        return array

    def points_screen(self, room_constrain_weak=None, boundary_constrain=None, points_on_blank=None):
        """根据邻接关系表筛选可放置点"""
        points_available = []
        if room_constrain_weak and boundary_constrain:  # 同时满足房间和边界两类约束
            # 得到room_names_need中在当前平面中实际存在的房间shapely形体
            rooms_need = []
            room_names_exist = list(
                set(room_constrain_weak).intersection(set(self.layout_info.columns)))  # 选取存在于户型中的房间名称
            for i in room_names_exist:
                room = self.poly_layout.loc["rec", i]
                rooms_need.append(room)
            rooms_need_union = shapely.ops.unary_union(rooms_need)

            for p in points_on_blank:
                point = geometry.Point(p[0], p[1])
                if point.touches(rooms_need_union) and point.touches(boundary_constrain):
                    points_available.append(list(p))

        elif not room_constrain_weak and boundary_constrain:
            for p in points_on_blank:
                point = geometry.Point(p[0], p[1])
                if point.touches(boundary_constrain):
                    points_available.append(list(p))

        elif room_constrain_weak and not boundary_constrain:
            rooms_need = []
            room_names_exist = list(
                set(room_constrain_weak).intersection(set(self.layout_info.columns)))  # 选取存在于户型中的房间名称
            for i in room_names_exist:
                room = self.poly_layout.loc["rec", i]
                rooms_need.append(room)
            rooms_need_union = shapely.ops.unary_union(rooms_need)

            for p in points_on_blank:
                point = geometry.Point(p[0], p[1])
                if point.touches(rooms_need_union):
                    points_available.append(list(p))

        elif not room_constrain_weak and not boundary_constrain:
            for p in points_on_blank:
                points_available.append(list(p))

        # 去除重复元素
        points_available_unique = []
        for item in points_available:
            if item not in points_available_unique:
                points_available_unique.append(item)
        return points_available_unique

    def points_screen_graph(self, points_available, graph_constrain):
        """筛选出潜在符合邻接关系的点，即某个点只要满足多个邻接关系约束中的一个，就被视为潜在符合"""
        if graph_constrain is None:
            return points_available
        else:
            points_available_new = []
            # 得到room_names_need中在当前平面中实际存在的房间shapely形体
            se_adjacent = graph_constrain.loc[:, self.new_room]
            se_adjacent_sub = se_adjacent[self.poly_layout_boundary_plus.columns]
            se_adjacent_names = se_adjacent_sub[se_adjacent_sub == 1].index

            if len(se_adjacent_names) != 0:
                for p in points_available:
                    point = geometry.Point(p[0], p[1])
                    for i in se_adjacent_names:
                        poly_room = self.poly_layout_boundary_plus.loc['rec', i]
                        if (point.touches(poly_room)) and (p not in points_available_new):
                            points_available_new.append(p)
            else:
                points_available_new = points_available
        return points_available_new

    # @timer_decorator
    def get_available_layout(self):
        layout = []
        # 获得当前可行的点
        deploy_points = self.get_deploy_points()
        if deploy_points is None or deploy_points.empty:
            return [], []
        # 获得当前需部署的房间的限制
        room_limits = room_sizes[self.new_room]
        for index in range(deploy_points.shape[1]):  # 遍历所有的可能的点
            x, y = deploy_points.iloc[0, index]
            quadrant = deploy_points.iloc[1, index]
            for w, d in room_limits:
                # 判断是否是二层，如果是则需符合下层墙体约束
                if self.layout_down is not None:
                    # 生成合适的shapely对象并判断是否是合适的选择
                    tmp_room = build_room(x, y, w, d, quadrant)
                    lis_corners = list(tmp_room.exterior.coords)

                    wall_restrict = True
                    for x_sub, y_sub in lis_corners:
                        if [x_sub, y_sub] not in self.lis_wall_points_down:
                            wall_restrict = False

                    if wall_restrict is True:
                        violation_judge = self.if_violation(tmp_room)
                        if not violation_judge:
                            layout.append(layout_change(x, y, w, d, quadrant))
                else:
                    # 生成合适的shapely对象并判断是否是合适的选择
                    tmp_room = build_room(x, y, w, d, quadrant)
                    violation_judge = self.if_violation(tmp_room)
                    if not violation_judge:
                        layout.append(layout_change(x, y, w, d, quadrant))
        return layout, deploy_points

    def find_diagonal_point(self, x, y, w, d, quadrant):
        """根据矩形point1信息和象限，找到对角线点"""
        if quadrant == 1:
            x1, y1 = x + w, y + d
        elif quadrant == 2:
            x1, y1 = x - w, y + d
        elif quadrant == 3:
            x1, y1 = x - w, y - d
        else:
            x1, y1 = x + w, y - d
        return x1, y1

    def get_point2_with_selected_point(self, se_point):
        """
        该函数可根据指定的点和象限找到所有可放置的对角线点
        :param se_point: 包含了点坐标和象限信息的Series
        :return:
        """
        lis_point2 = []
        # 获得当前可行的点
        deploy_points = se_point
        if (not isinstance(deploy_points, pd.Series)) or deploy_points.empty:
            print("输入数据格式有误")
            return []

        # 获得当前需部署的房间的限制
        room_limits = room_sizes[self.new_room]
        x, y = deploy_points["point"]
        quadrant = deploy_points["quadrant"]

        for w, d in room_limits:
            # 判断是否是二层，如果是则需符合下层墙体约束
            if self.layout_down is not None:
                # 生成合适的shapely对象并判断是否是合适的选择
                tmp_room = build_room(x, y, w, d, quadrant)
                lis_corners = list(tmp_room.exterior.coords)

                wall_restrict = True
                for x_sub, y_sub in lis_corners:
                    if [x_sub, y_sub] not in self.lis_wall_points_down:
                        wall_restrict = False

                if wall_restrict is True:
                    violation_judge = self.if_violation(tmp_room)
                    if not violation_judge:
                        # 计算point2坐标
                        x1, y1 = self.find_diagonal_point(x, y, w, d, quadrant)
                        lis_point2.append([x1, y1])
            else:
                # 生成合适的shapely对象并判断是否是合适的选择
                tmp_room = build_room(x, y, w, d, quadrant)
                violation_judge = self.if_violation(tmp_room)
                if not violation_judge:
                    # 计算point2坐标
                    x1, y1 = self.find_diagonal_point(x, y, w, d, quadrant)
                    lis_point2.append([x1, y1])
        return lis_point2

    def get_available_layout_fixed_point(self, se_point):
        """
        该函数可根据指定的点和象限找到所有可放置的矩形
        :param se_point: 包含了点坐标和象限信息的Series
        :return:
        """
        layout = []
        # 获得当前可行的点
        deploy_points = se_point
        if (not isinstance(deploy_points, pd.Series)) or deploy_points.empty:
            print("输入数据格式有误")
            return []

        # 获得当前需部署的房间的限制
        room_limits = room_sizes[self.new_room]
        x, y = deploy_points["point"]
        quadrant = deploy_points["quadrant"]

        for w, d in room_limits:
            # 判断是否是二层，如果是则需符合下层墙体约束
            if (self.layout_down is not None) and (not self.new_room.endswith('sub')):
                # 生成合适的shapely对象并判断是否是合适的选择
                tmp_room = build_room(x, y, w, d, quadrant)
                lis_corners = list(tmp_room.exterior.coords)

                wall_restrict = True
                for x_sub, y_sub in lis_corners:
                    if [x_sub, y_sub] not in self.lis_wall_points_down:
                        wall_restrict = False

                if wall_restrict is True:
                    violation_judge = self.if_violation(tmp_room)
                    if not violation_judge:
                        layout.append(layout_change(x, y, w, d, quadrant))

            else:
                # 生成合适的shapely对象并判断是否是合适的选择
                tmp_room = build_room(x, y, w, d, quadrant)
                violation_judge = self.if_violation(tmp_room)
                if not violation_judge:
                    layout.append(layout_change(x, y, w, d, quadrant))
        return layout

    def get_path(self):
        """得到包含客厅、餐厅的公共空间"""
        without_boundary = self.poly_layout.drop(columns=['boundary']).copy()
        rooms_as_path = list(['living', 'dining'])
        pub_rooms = list(set(self.poly_layout.columns.tolist()).intersection(set(rooms_as_path)))  # 筛选出存在的可做过道的房间
        without_living_dining = without_boundary.drop(columns = pub_rooms).copy()
        path_all = self.poly_layout.loc['rec', 'boundary'].difference(ops.unary_union(without_living_dining.values))
        return path_all

    # 判断当前新布置的房间是否违反了强约束规则
    def if_violation(self, room):
        """room 为shapely对象"""
        violation = False
        if room.area == 0:
            return True

        # 判断是否符合graph邻接关系约束
        if self.control is not None:
            se_adjacent = self.control.loc[:, self.new_room]
            se_adjacent_sub = se_adjacent[self.poly_layout_boundary_plus.columns]
            se_adjacent_names = se_adjacent_sub[se_adjacent_sub == 1].index

            if len(se_adjacent_names) > 0:
                for i in se_adjacent_names:
                    # print(i)
                    poly_room = self.poly_layout_boundary_plus.loc['rec', i]
                    if not room.intersects(poly_room):
                        return True

        # 判断是否符合RoomLocation位置要求
        # if not self.room_loc.empty:
        #     location_point_now = self.room_loc[self.new_room].values[0]
        #     if not room.contains(location_point_now):
        #         violation = True
        #         return violation

        # 判断boundary是否有效（boundary必须为连续形状，不能被切成多个）
        # if self.poly_blank_env.geom_type == "MultiPolygon":
        #     violation = True
        #     return violation

        # 判断房间是否超界boundary
        if self.new_room not in self.names_entrance:
            room_area_in_boundary = self.poly_blank_env_without_white_m.intersection(room).area
        else:
            room_area_in_boundary = room.area + 10000000000  # self.names_entrance中的对象不剪枝
        if room_area_in_boundary < (room.area - 1):
            return True

        # 如果当前的房间和之前的房间重叠面积太大，也不行。但嵌套房间此条不适用
        if (self.room_info is not None) and (self.new_room not in self.names_entrance):
            if not self.new_room.endswith("sub"):  # 非附属用房
                area_min_new_room = area_limits[self.new_room][0] * 1000000
                room_diff2 = room.difference(self.rooms_env_deployed_union)
                if room_diff2.area < (area_min_new_room * 0.8):  # 裁切后的房间面积不能小于房间最小面积要求的80%
                    return True
                if room_diff2.area < (room.area * 0.7):  # 裁切后的房间不能小于原房间面积的70%
                    return True
            else:  # 附属用房（主卫）
                area_min_new_room = area_limits[self.new_room][0] * 1000000
                rooms_deployed_union1 = self.rooms_env_deployed_union.difference(self.poly_layout.loc["rec", "room1"])  # 刨去主房间
                room_diff2 = room.difference(rooms_deployed_union1)
                if room_diff2.area < (area_min_new_room * 0.8):  # 裁切后的房间是否仍然满足房间最小面积要求的80%
                    return True

                if room_diff2.area / room.area < 0.6:  # 相交面积不能大于原房间面积的60%
                    return True

        # 实体房间不能挡大门和主入口
        if self.layout_down is None:  # 说明是一层
            names_as_entrance = ["entrance", 'entrance_sub']
        else:  # 说明是二层及以上
            names_as_entrance = ["entrance", 'entrance_sub', 'staircase']

        if self.new_room not in ["living", "dining", "hallway"]:
            for en in names_as_entrance:
                if en in self.poly_layout.columns:
                    entrance = self.poly_layout.loc['rec',  en]
                    poly_blank_sub = self.poly_blank_env_without_white_m.difference(entrance)
                    entrance_line = entrance.exterior
                    coline_entrance = entrance_line.intersection(poly_blank_sub)  # 主入口理论可开门线
                    rooms_entity_ = ops.unary_union(self.lis_room_entity + [room])
                    # self.poly_visualization(polys=self.lis_room_entity)
                    base_intersection = coline_entrance.difference(rooms_entity_)  # 房间与入口理论可开门线的差集
                    if base_intersection.length < 900:  # 可开门尺寸不足
                        return True

        # 判断白体m是否孤立
        if self.new_room in ["white_m1", "white_m2", "white_m3"]:
            co_line = []
            for w in ["white_south", "white_north", "white_east", "white_west", "white_m1", "white_m2", "white_m3"]:
                if (w != self.new_room) and (w in self.poly_layout.columns):
                    cord1 = room.exterior.coords
                    cord2 = self.poly_layout.loc["rec", w].exterior.coords
                    line1 = geometry.LineString(np.array(cord1))
                    line2 = geometry.LineString(np.array(cord2))
                    shared_line = ops.shared_paths(line1, line2).geoms
                    if len(shared_line) != 0:
                        for line in shared_line:
                            length = line.length
                            co_line.append(length)
                            if length > 0:
                                break
                if sum(co_line) > 0:
                    break
            if sum(co_line) == 0:
                return True

        # 判断是否满足空间关系中的强制约束
        # columns_without_boundary = self.room_adjacent_matrix.columns.drop("boundary")
        # matrix_relative = self.room_adjacent_matrix.loc[self.new_room, columns_without_boundary]
        # placeholder = matrix_relative[matrix_relative == 1].index.tolist()  # 筛选强约束房间名称
        #
        # if len(placeholder) != 0:
        #     for item in placeholder:
        #         if item in self.poly_layout.columns:
        #             poly_room_strong_constrain = self.poly_layout.loc["rec", item]
        #             if not room.intersects(poly_room_strong_constrain):  # 判断是否与强约束房间相交
        #                 violation = True
        #                 return violation

        # 判断主要房间是否有足够采光
        # print(self.room_adjacent_matrix)
        adj_values = self.room_adjacent_matrix.loc[self.new_room, self.light_names].values
        if 1 in adj_values or -1 in adj_values:
            # 构建所有白体shapely集合
            poly_light = self.poly_layout.loc["rec", self.light_names]
            light_union = ops.unary_union(poly_light)  # 合并所有白体

            # 判断白体个数
            if light_union.geom_type == "Polygon":
                iteration_light = [light_union]
            else:
                iteration_light = light_union.geoms
            # 房间与所有白体共线的长度
            co_line = []
            for iter_light in iteration_light:
                coord_lr = room.exterior.coords
                line_lr = geometry.LineString(np.array(coord_lr))
                shared_line = line_lr.intersection(iter_light)
                if shared_line.length > 0:
                    # for line_item in shared_line:
                    co_line.append(shared_line.length)

            # 判断共线长度
            line_length = sum(co_line)
            if self.new_room in window_limits.index:
                if line_length < window_limits[self.new_room]:
                    return True

        return violation

    def poly_visualization(self, polys):
        """可视化多边形，debug用"""
        if isinstance(polys, list):
            for poly in polys:
                x = [p[0] for p in list(poly.exterior.coords)]
                y = [p[1] for p in list(poly.exterior.coords)]
                plt.plot(x, y, color='black')
                plt.fill(x, y, color='lightgray', alpha=0.8)
            plt.show()
        elif isinstance(polys, shapely.geometry.Polygon):
            x = [p[0] for p in list(polys.exterior.coords)]
            y = [p[1] for p in list(polys.exterior.coords)]
            plt.plot(x, y, color='black')
            plt.fill(x, y, color='lightgray', alpha=0.8)
            plt.show()

        elif isinstance(polys, shapely.geometry.MultiPolygon):
            for poly in polys.geoms:
                x = [p[0] for p in list(poly.exterior.coords)]
                y = [p[1] for p in list(poly.exterior.coords)]
                plt.plot(x, y, color='black')
                plt.fill(x, y, color='lightgray', alpha=0.8)
            plt.show()


class GraphOriginal:
    def __init__(self, layout_info, points, file_name, path_out):
        self.points = points
        self.layout_info = layout_info
        self.path_out = path_out
        self.file_name = file_name
        self.text_size = 18
        self.text_size_sub = 14

        self.layout_info.index = ["x", "y", "w", "d"]
        wrong_names = ["dinning", "dinner", "toilet_main", "toilet2", "toilet3", "toilet4"]
        wrong_name_trans = {
            "dinning": "dining", "dinner": "dining",
            "toilet_main": "bath1", "toilet2": "bath2", "toilet3": "bath3", "toilet4": "bath4",
        }
        for i in self.layout_info.columns:
            if i in wrong_names:
                self.layout_info = self.layout_info.rename(columns={i: wrong_name_trans[i]})

        # 平面布置图
        self.graph_color = {
            "boundary": "#FFE27D", "entrance": "whitesmoke", "hallway": "#FFE27D",
            "white_south": "whitesmoke", "white_north": "whitesmoke", "white_east": "whitesmoke",
            "white_west": "whitesmoke", 'white_third': 'whitesmoke', 'white_fourth':'whitesmoke',
            "white_m1": "whitesmoke", "white_m2": "whitesmoke", "white_m3": "whitesmoke", "white_m4": "whitesmoke",
            "black1": "darkgrey", "black2": "darkgrey", "black3": "darkgrey", "black4": "darkgrey",
            "room1": "#F3A169", "room2": "#F3A169", "room3": "#F3A169", "room4": "lightgray",
            "living": "#FFE27D", "dining": "#C5D5AE",
            "bath1": "#6483A6", "bath2": "#90AEC4", "bath1_sub": "#90AEC4", "bath2_sub": "#90AEC4",
            "kitchen": "#80A6AF", "storeroom": "#90AEC4",
            "gate": "#FFE27D", "courtyard": "whitesmoke", "staircase": "#8474A0", "porch": "whitesmoke",
            "study_room": "#F3A169",
            "blank": "whitesmoke", "entrance_sub": "whitesmoke", "garage": "gray"
        }

        self.text_trans = {
            "entrance": "Entrance", "entrance_sub": "Entrance_sub", "hallway": "hallway",
            "white_south": "Light", "white_north": "Light", "white_east": "Light", "white_west": "Light",
            'white_third': 'Light', 'white_fourth': 'Light',
            "white_m1": "Light", "white_m2": "Light", "white_m3": "Light", "white_m4": "Light",
            "black1": "", "black2": "", "black3": "", "black4": "",
            "room1": "Room1", "room2": "Room2", "room3": "Room3", "room4": "Room4", "study_room": "Study",
            "living": "Living", "dining": "Dining", "bath1": "Bath1", "bath2": "Bath2", "bath1_sub": "Bath_sub1",
            "bath2_sub": "Bath_sub2",
            "kitchen": "Kitchen", "staircase": "Stair",
            "storeroom": "Storeroom", "pub": "", "gate": "Gate", "courtyard": "Courtyard", "porch": "Porch",
            "blank": "Blank", "garage": "Garage"
        }

        # self.text_trans = {
        #     "entrance": "入口",
        #     "white_south": "采光", "white_north": "采光", "white_third": "采光",
        #     "white_m1": "采光", "white_m2": "采光", "white_m3": "采光", "black1": " ", "black2": " ", "black3": " ",
        #     "room1": "卧室", "room2": "卧室", "room3": "其它",
        #     "living": "客厅", "dining": "餐厅", "toilet1": "卫", "toilet2": "主卫", "kitchen": "厨", "storage": "储藏"
        # }

    def draw_plan(self):
        # mpl.rcParams["font.sans-serif"] = ["simsun"]  # 指定默认字体
        mpl.rcParams["font.sans-serif"] = ["Times New Roman"]  # 指定默认字体
        mpl.rcParams["axes.unicode_minus"] = False  # 解决保存图像是负号"-"显示为方块的问题

        fig = plt.figure("layout", figsize=(9, 9))
        ax1 = plt.subplot(1, 1, 1)
        plt.ion()  # 开启交互模式
        plt.cla()

        name_list = self.layout_info.columns.values.tolist()
        lis_exclude = []

        if "entrance" in name_list:
            name_list.remove("entrance")
            name_list = name_list + ["entrance"]

        for i, item in enumerate(name_list):
            tmp = self.layout_info[item]
            X, Y, W, H = tmp.values
            if item not in lis_exclude:
                rect = plt.Rectangle((X, Y), W, H, fill=True, facecolor=self.graph_color[item], alpha=0.8, lw=3)
                ax1.add_patch(rect)
                rect1 = plt.Rectangle((X, Y), W, H, fill=False, edgecolor="black", alpha=0.9, lw=3)
                ax1.add_patch(rect1)
                rect.set_label(item)

            # if item not in ["white_m1", "whtie_m2", "white_m3"]:
            if item != "boundary":
                plt.text(
                    X + W / 2, Y + H / 2,
                    s="%s" % (self.text_trans[item]),
                    color="black",
                    verticalalignment="center",
                    horizontalalignment="center",
                    size=self.text_size
                )

        plt.xlim([-3000, 18000])
        plt.ylim([-3000, 18000])

        px = []
        py = []

        for p in self.points:
            px.append(p[0])
            py.append(p[1])
        plt.scatter(px, py, s=40, c="r")

        plt.tight_layout(pad=0.05)
        fig.subplots_adjust(wspace=0.2, hspace=0.2)
        plt.show()
        plt.pause(0.5)
        plt.savefig(self.path_out + str(self.file_name) + ".jpeg", dpi=300)
        plt.ioff()
        plt.clf()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pylab as mpl
    from graph_maker import Transfer2Graph

    # path = "data_test/"
    path = "C:/Users/SHU/Desktop/2025_06_13_22_02_11/error_log/rural/"
    file_name = "rural_floor2_B-L-25.xlsx"

    excel_file = pd.ExcelFile(path + file_name)
    sheet_names = excel_file.sheet_names

    df = pd.read_excel(path + file_name, index_col=0, sheet_name='floor2')
    df_down = pd.read_excel(path + file_name, index_col=0, sheet_name='floor1')

    env_names = [
        'boundary',
        'entrance',
        'entrance_sub',
        'white_south',
        'white_north',
        'white_west',
        'white_east',
        'black1',
        'black2',
        'black3',
        'black4',
        'staircase',
    ]

    room_names = [
        'white_m1',
        'white_m2',
        'white_m3',
        'white_m4',
        'living',
        # 'room1',
        'room2',
        'room3',
        'room4',
        'kitchen',
        'storeroom',
        'bath1',
        'bath2',
        'bath1_sub',
        'dining',
    ]

    env_name_exist = [i for i in df.columns if i in env_names]
    room_name_exist = [i for i in df.columns if i in room_names]
    env_info = df[env_name_exist].copy()

    room_info = df[room_name_exist].copy()
    room_name = "room1"

    node = get_available_layout(
        new_room_name=f"{room_name}",
        env_info=env_info,
        room_info=room_info,
        layout_down=df_down,
        control=None
    )

    leaf_node, deployed_points = node.get_available_layout()
    points_lis = deployed_points.loc["point", :].values
    counter = 0
    for m in leaf_node:
        counter += 1

    for i in range(len(leaf_node)):
        x, y, w, d = leaf_node[i]
        room_tmp = np.array([x, y, w, d])
        room_info_tmp = pd.DataFrame(
            room_tmp.T,
            columns=[f"{room_name}"],
            index=["x", "y", "w", "d"]
        )
        counter += len(room_info_tmp.columns)
        room_info_total = pd.concat([env_info, room_info, room_info_tmp], axis=1)
        case = GraphOriginal(layout_info=room_info_total, points=points_lis, file_name="lala.jpeg", path_out="")
        case.draw_plan()
