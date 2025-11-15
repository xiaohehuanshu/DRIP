import pandas as pd
import shapely
import time
from tools_layout_modeling import layout2geopandas, adjacent_matrix_shapely, shape_parm_calibration
from .evaluator_room_shape import RoomShapeScore
from .evaluator_room_light import RoomLightScore
from .evaluator_path_shape import PathShapeScore
from .evaluator_circulation import circulation_evaluator
from .evaluator_others import room_number_score, intersection_area
from .evaluator_graph_similarity import graph_similarity_calculator
from utils.graph_layout import GraphPlan

# from evaluator_graph_similarity import graph_similarity_calculator

criterion_shape = pd.read_excel("./evaluator/criterion_setting_file/shape_rooms.xlsx", index_col=0)
criterion_light = pd.read_excel("./evaluator/criterion_setting_file/relation_light.xlsx", index_col=0)
criterion_win = pd.read_excel("./evaluator/criterion_setting_file/shape_windows.xlsx", index_col=0).dropna(axis=1, how="any")
circulation_absolute = pd.read_excel("./evaluator/criterion_setting_file/relation_circulation_absolute.xlsx", index_col=0)
circulation_relative = pd.read_excel("./evaluator/criterion_setting_file/relation_circulation_relative.xlsx", index_col=0)
graph_weights = pd.read_excel("./evaluator/criterion_setting_file/graph_similarity.xlsx", index_col=0)


def layout_evaluator(df_info, room_names_all, mode, graph=None):
    # suit the floor number
    if ('staircase' in df_info.columns) and ('entrance' not in df_info.columns):
        df_info = df_info.rename(columns={'staircase': 'entrance'})

    # transfer the variables
    df_shapes = layout2geopandas(layout_info=df_info)  # 得到geopandas房间对象
    adj_matrix = adjacent_matrix_shapely(df_shapely=df_shapes)  # 得到邻接矩阵

    # calculate the room shape score
    criterion_trans = shape_parm_calibration(df_info=df_info, df_parm=criterion_shape.copy(), room_need=room_names_all)  # 校准房间尺寸超参数
    case_shape = RoomShapeScore(df_shape=df_shapes, df_info=df_info, adj_matrix=adj_matrix, criterion=criterion_trans)
    score_shape = case_shape.room_shape_total_score()

    # renew df_shape variable
    shapes_real = case_shape.df_shape
    col_rooms = shapes_real.columns
    df_shapes.loc['poly', col_rooms] = shapes_real.loc['poly', col_rooms].values

    # calculate daylighting
    room_names_need = [i for i in room_names_all if not i.startswith("white")]
    case_light = RoomLightScore(
        df_shape=df_shapes,
        df_info=df_info,
        criterion=criterion_light,
        criterion_win=criterion_win,
        room_names_need=room_names_need,
    )
    score_light = case_light.room_light_total_score()

    # calculate the path shape score
    case_path = PathShapeScore(df_shape=df_shapes)
    score_path_other, score_path_access = case_path.path_shape_total_score()

    # calculate the score of living and dining relation
    score_living_dining = case_path.living_dining_unit_score()

    # calculate the circulation score
    df_shape_rooms = case_shape.df_shape
    df_shape_rooms, names_to_del = del_multipolygon(df_poly=df_shape_rooms)  # 删除multipolygon房间
    df_shape_rooms_all = df_shapes.copy()  # 将circulation_evaluator传入的全部房间数据更新为最新数据
    for r in names_to_del:
        df_shape_rooms_all = df_shape_rooms_all.drop(r, axis=1)  # 删除 multipolygon的房间
    score_circulation = circulation_evaluator(
        gpd_rooms=df_shape_rooms,
        gpd_all=df_shape_rooms_all,
        df_weight_absolute=circulation_absolute,
        df_weight_relative=circulation_relative,
    )
    # calculate the room number socre
    score_room_num = room_number_score(rooms_all=room_names_all, rooms_actual=df_info.columns.tolist())

    # calculate the score of the overlap of rooms
    score_overlap = intersection_area(df_shape=df_shapes)

    # calculate the graph similarity between the inquire and output results
    score_adjacent = graph_similarity_calculator(
        df_adj_target=graph, layout_info=df_info, weights=graph_weights
    )

    # calculate the rule-based final score
    score_dic = {
        "room_number": score_room_num * 1.1,
        "lighting": score_light * 1.1,
        "circulation": score_circulation * 1,
        "shape_room": score_shape * 0.9,
        "path_others": score_path_other * 0.5,
        "path_access": score_path_access * 0.5,
        "relation_liv_din": score_living_dining * 0.1,
        "overlap": score_overlap * 0.1,
        'room_adjacent': score_adjacent * 1.2
    }

    score_final = sum(score_dic.values())

    if mode == "test":
        case_draw = GraphPlan(layout_info=df_shapes, file_name="floor_plan_ache", path_out="", if_save=True)
        case_draw.draw_plan()
    return score_final, score_dic


def del_multipolygon(df_poly):
    # 排除掉multipolygon的情况
    recorder = []
    for c in df_poly.columns:
        poly = df_poly.loc["poly", c]
        if poly.geom_type == "MultiPolygon":
            df_poly = df_poly.drop(c, axis=1)
            recorder.append(c)
    return df_poly, recorder


if __name__ == "__main__":
    import os

    rooms_need = ["room1", "room2", "room3", "room4", "living", "kitchen", "bath1", "bath2", "dining"]
    lis = ['rural_floor1_E-I-176.xlsx']
    path = "C:/Users/SHU/Desktop/2025_05_19_22_00_59/results/rural/"
    reward = 0
    for root, dirs, files in os.walk(path):
        # files.reverse()
        for i, file in enumerate(files):
            if (file not in lis) and (i >= 353):
                print(i, file)
                df_info = pd.read_excel(path + file, index_col=0, sheet_name="floor1")

                # df_info = floor_number_adapter(df=df_info, floor_num=0)
                # df_shapes = layout2geopandas(layout_info=df_info)
                # df_matrix = adjacent_matrix_shapely(df_shapely=df_shapes)
                # print(df_matrix)
                score_all, _ = layout_evaluator(
                    df_info=df_info,
                    room_names_all=rooms_need,
                    mode="train",
                )
                reward += score_all
                print(score_all)
                print("-----------------")
    print("Benchmark reward:", f"{reward:.2f}")
