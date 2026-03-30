import pandas as pd
import numpy as np
from tools_layout_modeling import layout2geopandas, adjacent_matrix_shapely_GAT

# 设置 Pandas 显示选项，确保打印时不省略内容
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 自动调整显示宽度
pd.set_option('display.max_colwidth', None)  # 显示每列的最大宽度


def jaccard_similarity(A_input, A_output):
    """
    计算两个二元邻接矩阵之间的 Jaccard 相似度。

    参数:
        A_input (np.ndarray): 输入图的邻接矩阵（二元，0 或 1）
        A_output (np.ndarray): 输出图的邻接矩阵（二元，0 或 1），目标图

    返回:
        float: Jaccard 相似度值，范围 [0, 1]
    """
    # 确保两个邻接矩阵维度一致
    assert A_input.shape == A_output.shape, "邻接矩阵维度不一致"

    # 计算交集和并集的边数
    intersection = np.sum((A_input == 1) & (A_output == 1))
    union = np.sum((A_output == 1))

    # 防止除以零
    if union == 0:
        return 1.0  # 如果两个图都为空，则视为完全匹配

    # 返回 Jaccard 相似度
    return intersection / union

def graph_jaccard_reward(A_input, A_output, alpha=1.0):
    """
    将 Jaccard 相似度转换为奖励值。

    参数:
        A_input (np.ndarray): 输入图的邻接矩阵（二元）
        A_output (np.ndarray): 输出图的邻接矩阵（二元）
        alpha (float): 奖励缩放因子（默认 1.0）

    返回:
        float: 图相似度奖励值
    """
    similarity = jaccard_similarity(A_input, A_output)
    return alpha * similarity


class Transfer2GraphEdges:
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
        self.graph_control = None

    def add_direction_cubes(self):
        x, y, w, d = self.layout_ori.loc[:, 'boundary']
        cube_west = [-1200, 0, 1200, d]
        cube_east = [w, 0, 1200, d]
        cube_south = [0, -1200, w, 1200]
        cube_north = [0, d, w, 1200]
        add_columns = ['boundary_west', 'boundary_east', 'boundary_south', 'boundary_north']
        self.layout_ori.loc[:, add_columns] = np.array([cube_west, cube_east, cube_south, cube_north]).T
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


def graph_similarity_calculator(df_adj_target, layout_info):
    """
    :param df_adj_former: 目标图
    :param layout_info_latter: 户型矢量参数
    :param weights: 权重
    """
    if df_adj_target is None:
        return 0
    else:
        case = Transfer2GraphEdges(layout_ori=layout_info)
        df_adj_now = case.trans_graph_matrix()

        # 将满足条件的元素归零
        # mask = np.random.rand(*df_adj_now.shape) < 0.9
        # df_adj_now = df_adj_now.mask(mask, 0)
        score_adj = graph_jaccard_reward(A_input=df_adj_now.values, A_output=df_adj_target.values, alpha=1.0)
    return score_adj


if __name__ == "__main__":
    import numpy as np
    # 获取邻接矩阵权重值
    # 输入参考户型数据，并转化为矩阵（待更改输入接口）
    path_data = 'D:\\ONGOING\\RL_house\\dataset\\data6000_graph1\\city\\'
    file = 'city_large_A2-1-无-2.xlsx'
    layout_info_former = pd.read_excel(path_data + file, sheet_name='floor1', index_col=0)
    df_adj_tar = pd.read_excel(path_data + file, sheet_name='floor1_graph', index_col=0)

    case1 = Transfer2GraphEdges(layout_ori=layout_info_former)
    df_adj_now1 = case1.trans_graph_matrix()

    score = graph_similarity_calculator(df_adj_target=df_adj_now1, layout_info=layout_info_former)
    print(score)

















