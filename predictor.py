import pandas as pd
import torch
import numpy as np
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import csr_matrix
from tools_layout_modeling import layout2geopandas, adjacent_matrix_shapely
from torch_geometric.data import Data, Batch, Dataset


class Transfer2Graph:
    def __init__(self, layout_ori=None):
        self.layout_ori = layout_ori.copy()
        self.names_env = [
            'entrance', 'entrance_sub',
            'boundary_west', 'boundary_east', 'boundary_south', 'boundary_north',
            'white_south', 'white_north', 'white_east', 'white_west',
            'black1', 'black2', 'black3', 'black4',
        ]
        self.names_room = [
            'white_m1', 'white_m2', 'white_m3', 'white_m4',
            'garage', 'room1', 'room2', 'living', 'room3', 'room4', 'study_room', 'kitchen', 'staircase',
            'bath1', 'bath2', 'bath1_sub', 'storeroom', 'hallway', 'dining'
        ]

        self.super_parm = 1
        self.room_names_all = self.names_env + self.names_room
        self.add_direction_cubes()

    def add_direction_cubes(self):
        x, y, w, d = self.layout_ori.loc[:, 'boundary']
        cube_west = [-1200, 0, 1200, d]
        cube_east = [w, 0, 1200, d]
        cube_south = [0, -1200, w, 1200]
        cube_north = [0, d, w, 1200]
        add_columns = ['boundary_west', 'boundary_east', 'boundary_south', 'boundary_north']
        self.layout_ori[add_columns] = np.array([cube_west, cube_east, cube_south, cube_north]).T


    def trans_input_matrix(self):
        df_zero = pd.DataFrame(np.zeros((4, len(self.room_names_all))),
                               index=['x', 'y', 'w', 'd'],
                               columns=self.room_names_all)
        columns = [i for i in self.room_names_all if i in self.layout_ori.columns]
        df_info_now = self.layout_ori[columns]
        df_zero[df_info_now.columns] = df_info_now.values

        df_shape = layout2geopandas(layout_info=df_zero)
        df_adj = adjacent_matrix_shapely(df_shapely=df_shape)

        df_edges = df_adj.loc[self.room_names_all, self.room_names_all]
        adj_matrix_sparse = csr_matrix(df_edges.values)
        edge_index = from_scipy_sparse_matrix(adj_matrix_sparse)[0].detach().cpu().numpy().tolist()

        df_node = pd.DataFrame(np.zeros((1, len(self.room_names_all))), index=['room'], columns=self.room_names_all)
        for i in df_info_now.columns:
            df_node.loc['room', i] = 1
        node = df_node.values.T.tolist()

        data_graph = Data(x=torch.tensor(node), edge_index=torch.tensor(edge_index))
        return data_graph

    def markov_data_trans(self):
        graph_input = self.trans_input_matrix()
        return graph_input


def predict(layout):
    case = Transfer2Graph(layout_ori=layout)
    graph = case.markov_data_trans()
    return graph

