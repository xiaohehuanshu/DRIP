# -*- coding=utf-8 -*-
from typing import Any

import numpy as np
import pandas as pd
from tools_layout_modeling import build_room, layout2geopandas, adjacent_matrix_shapely_GAT

pd.set_option('expand_frame_repr', False)
np.set_printoptions(threshold=np.inf)
np.random.seed(2025)  # 设置种子

class Transfer2Graph:
    """
    从原始数据中提取输入数据，将矢量数据转化为像矩阵
    """
    def __init__(self, layout_ori=None):
        self.layout_ori = layout_ori

        self.names_env = [
            'entrance', 'entrance_sub',
            'boundary_west', 'boundary_east', 'boundary_south', 'boundary_north',
            'white_south', 'white_north', 'white_east', 'white_west',
            'black1', 'black2', 'black3', 'black4',
            'white_m1', 'white_m2', 'white_m3', 'white_m4',
        ]
        self.names_room = [
            'garage', 'room1', 'room2', 'living', 'room3', 'room4', 'study_room', 'kitchen', 'staircase',
            'bath1', 'bath2', 'bath1_sub', 'storeroom', 'hallway', 'dining'
        ]

        self.room_names_all = self.names_env + self.names_room  # 本算法支持的全部房间名称
        self.add_direction_cubes()

    def add_direction_cubes(self):
        x, y, w, d = self.layout_ori.loc[:, 'boundary']
        cube_west = [-1200, 0, 1200, d]
        cube_east = [w, 0, 1200, d]
        cube_south = [0, -1200, w, 1200]
        cube_north = [0, d, w, 1200]
        add_columns = ['boundary_west', 'boundary_east', 'boundary_south', 'boundary_north']
        self.layout_ori[add_columns] = np.array([cube_west, cube_east, cube_south, cube_north]).T
        # print(self.layout_ori)

    def trans_graph_matrix(self):
        """
        :param df_info_now: 作为输入数据的当前平面信息，包括环境和已布置房间
        :param names_need_all: 需要生成的全部房间，包括环境和全部内部房间
        """
        df_zero = pd.DataFrame(np.zeros((4, len(self.room_names_all))),
                               index=['x', 'y', 'w', 'd'],
                               columns=self.room_names_all)  # 19个通道
        columns = [i for i in self.room_names_all if i in self.layout_ori.columns]
        df_info_now = self.layout_ori[columns]  # 给df_info列重新排序
        df_zero[df_info_now.columns] = df_info_now.values  # 得到全房间属性信息

        # 得到边数据
        df_shape = layout2geopandas(layout_info=df_zero)
        df_adj = adjacent_matrix_shapely_GAT(df_shapely=df_shape)
        df_edges = df_adj.loc[self.room_names_all, self.room_names_all]  # 规范房间顺序
        return df_edges

    def trans_input_graph_partly(self):
        """
        :param df_info_now: 作为输入数据的当前平面信息，包括环境和已布置房间
        :param names_need_all: 需要生成的全部房间，包括环境和全部内部房间
        """
        super_parameter = 0.7
        df_zero = pd.DataFrame(np.zeros((4, len(self.room_names_all))),
                               index=['x', 'y', 'w', 'd'],
                               columns=self.room_names_all)  # 19个通道

        columns = [i for i in self.room_names_all if i in self.layout_ori.columns]
        df_info_now = self.layout_ori[columns]  # 给df_info列重新排序
        df_zero[df_info_now.columns] = df_info_now.values  # 得到全房间属性信息

        # 得到边数据
        df_shape = layout2geopandas(layout_info=df_zero)
        df_adj = adjacent_matrix_shapely_GAT(df_shapely=df_shape)
        df_edges = df_adj.loc[self.room_names_all, self.room_names_all]  # 规范房间顺序

        # 得到随机归零数据
        if np.random.rand(1) < 0.3:  # 30%比例不输入图，即输入的图全部为0
            for i, col in enumerate(df_edges.columns):
                for idx in df_edges.index[:i]:
                    df_edges.loc[idx, col] = 0
                    df_edges.loc[col, idx] = 0
        else:
            for i, col in enumerate(df_edges.columns):
                for idx in df_edges.index[:i]:
                    if col in self.names_room:
                        value = df_edges.loc[idx, col]
                        if value == 1:
                            if np.random.rand(1) < super_parameter:  # 按照一定比例将一部分数值归0
                                df_edges.loc[idx, col] = 0
                                df_edges.loc[col, idx] = 0
        return df_edges


if __name__ == "__main__":
    import os
    import pandas as pd

    path_in = 'D:/ONGOING/RL_house/dataset/data6000/'

    for root, dirs, files in os.walk(path_in):
        for file in files:
            print(file)
            path = os.path.join(root, file)
            excel_file = pd.ExcelFile(path)
            sheet_names = excel_file.sheet_names

            dic = {}
            for i, sheet in enumerate(sheet_names):
                df = pd.read_excel(path, index_col=0, sheet_name=sheet)
                case = Transfer2Graph(layout_ori=df)
                edges = case.trans_input_graph_partly()

                if sheet.startswith('Sheet'):
                    sheet = 'floor' + str(i + 1)

                dic[sheet+'_graph'] = edges
                dic[sheet] = df

            path_new = path.replace("data6000", "data6000_graph")
            with pd.ExcelWriter(path_new, engine='openpyxl') as writer:
                for sheet_name, df_out in dic.items():
                    df_out.to_excel(writer, sheet_name=sheet_name, index=True)










