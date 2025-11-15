import pandas as pd
import numpy as np
from tools_layout_modeling import layout2geopandas, adjacent_matrix_shapely_GAT

# 设置 Pandas 显示选项，确保打印时不省略内容
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 自动调整显示宽度
pd.set_option('display.max_colwidth', None)  # 显示每列的最大宽度


def add_direction_cubes(layout):
    """将layout_info改成适配graph的数据格式"""
    x, y, w, d = layout.loc[:, 'boundary']
    cube_west = [x - 1200, y, 1200, d]
    cube_east = [x + w, y, 1200, d]
    cube_south = [x, y - 1200, w, 1200]
    cube_north = [x, y + d, w, 1200]
    layout = layout.drop('boundary', axis=1)  # 减去boundary
    add_columns = ['boundary_west', 'boundary_east', 'boundary_south', 'boundary_north', ]
    layout[add_columns] = np.array([cube_west, cube_east, cube_south, cube_north]).T
    # print(self.layout_ori)
    return layout

def graph_similarity_calculator(df_adj_target, layout_info, weights):
    """
    :param df_adj_former: 目标图
    :param layout_info_latter: 户型矢量参数
    :param weights: 权重
    """
    if df_adj_target is None:
        return 0
    else:
        layout_add = add_direction_cubes(layout=layout_info)
        df_shapely_latter = layout2geopandas(layout_add)
        df_adj_now = adjacent_matrix_shapely_GAT(df_shapely_latter)
        score_adj = 0
        counter_adj = 0

        adj_headers = df_adj_now.columns.tolist()
        for idx_i, i in enumerate(adj_headers):  # 行
            for idx_j, j in enumerate(adj_headers[:idx_i]):  # 列
                if (i in weights.columns) and (j in weights.columns):
                    some_adj_target = df_adj_target.loc[i, j]
                    some_adj_now = df_adj_now.loc[i, j]
                    some_weights = weights.loc[i, j]
                    if some_adj_target == 1:
                        counter_adj += some_weights
                        if (some_adj_now == 1) and (some_weights > 0):
                            score_adj += some_weights
        if counter_adj == 0:
            return 0
        score_adj = score_adj / counter_adj
    return score_adj


if __name__ == "__main__":
    # 获取邻接矩阵权重值
    path = 'criterion_setting_file/graph_similarity.xlsx'
    weights = pd.read_excel(path, index_col=0)
    # 输入参考户型数据，并转化为矩阵（待更改输入接口）
    path_data = 'C:/Users/SHU/Desktop/2025_05_19_22_00_59/results/rural/'
    file = 'rural_floor1_B-L-134.xlsx'
    layout_info_former = pd.read_excel(path_data + file, sheet_name='floor1', index_col=0)
    df_adj_tar = pd.read_excel(path_data + file, sheet_name='floor1_graph', index_col=0)

    score = graph_similarity_calculator(df_adj_target=df_adj_tar, layout_info=layout_info_former, weights=weights)
    print(score)

















