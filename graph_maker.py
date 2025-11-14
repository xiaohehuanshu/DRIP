import numpy as np
import pandas as pd
from tools_layout_modeling import build_room, layout2geopandas, adjacent_matrix_shapely_GAT

pd.set_option('expand_frame_repr', False)
np.set_printoptions(threshold=np.inf)

class Transfer2GraphEdges:
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

        self.room_names_all = self.names_env + self.names_room
        self.add_direction_cubes()
        self.graph_control = None

    def add_direction_cubes(self):
        x, y, w, d = self.layout_ori.loc[:, 'boundary']
        cube_west = [-1200, 0, 1200, d]
        cube_east = [w, 0, 1200, d]
        cube_south = [0, -1200, w, 1200]
        cube_north = [0, d, w, 1200]
        add_columns = ['boundary_west', 'boundary_east', 'boundary_south', 'boundary_north']
        self.layout_ori.loc[:, add_columns] = np.array([cube_west, cube_east, cube_south, cube_north]).T

    def trans_graph_matrix(self):
        df_zero = pd.DataFrame(np.zeros((4, len(self.room_names_all))),
                               index=['x', 'y', 'w', 'd'],
                               columns=self.room_names_all)
        columns = [i for i in self.room_names_all if i in self.layout_ori.columns]
        df_info_now = self.layout_ori[columns]
        df_zero[df_info_now.columns] = df_info_now.values

        df_shape = layout2geopandas(layout_info=df_zero)
        df_adj = adjacent_matrix_shapely_GAT(df_shapely=df_shape)
        df_edges = df_adj.loc[self.room_names_all, self.room_names_all]
        return df_edges

    def trans_input_graph_partly(self, graph_control=False, random=False):
        super_parameter = 0.7
        df_info_all = pd.DataFrame(np.zeros((4, len(self.room_names_all))),
                               index=['x', 'y', 'w', 'd'],
                               columns=self.room_names_all)

        columns = [i for i in self.room_names_all if i in self.layout_ori.columns]
        df_info_now = self.layout_ori[columns]
        df_info_all[df_info_now.columns] = df_info_now.values

        if isinstance(graph_control, list):
            print('\n', graph_control)
            self.graph_control = graph_control
            df_edges = pd.DataFrame(
                np.zeros((len(self.room_names_all), len(self.room_names_all))),
                index=self.room_names_all,
                columns=self.room_names_all
            )
            for i, col in enumerate(df_edges.columns):
                for idx in df_edges.index:
                    if col == idx:
                        df_edges.loc[idx, col] = -1
            for key, value in self.graph_control:
                df_edges.loc[key, value] = 1
                df_edges.loc[value, key] = 1

        else:
            if random:
                df_shape = layout2geopandas(layout_info=df_info_all)
                df_adj = adjacent_matrix_shapely_GAT(df_shapely=df_shape)
                df_edges = df_adj.loc[self.room_names_all, self.room_names_all]

                if np.random.rand(1) < 0.3:
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
                                    if np.random.rand(1) < super_parameter:
                                        df_edges.loc[idx, col] = 0
                                        df_edges.loc[col, idx] = 0
            else:
                df_edges = pd.DataFrame(
                    np.zeros((len(self.room_names_all), len(self.room_names_all))),
                    index=self.room_names_all,
                    columns=self.room_names_all
                )

        return df_edges

