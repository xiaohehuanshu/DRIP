import traceback
import json
import numpy as np
import pandas as pd
import copy, os, time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from wcwidth import wcswidth

plt.rcParams["font.family"] = ["sans-serif", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

import torch
from torch_geometric.data import Batch
from shapely import ops
from shapely.geometry import Point, MultiPoint, Polygon
from model import resnet_encoder, resnet_decoder_a, resnet_decoder_b, GAT
from tools_layout_modeling import layout2geopandas
from evaluator.evaluator_main import layout_evaluator
from predictor import Transfer2Graph
from pruning.get_leaf_node_edges import get_available_layout
from graph_maker import Transfer2GraphEdges

# from pruning.get_leaf_node_points import get_available_layout
from utils.graph_layout import GraphPlan
from post_processer import PostProcess

# Optional imports for QwenVL baseline inference
try:
    from PIL import Image
    from utils.make_dataset import generate_prompt_from_file
    QWENVL_UTILS_AVAILABLE = True
except ImportError:
    QWENVL_UTILS_AVAILABLE = False


class LayoutGenerator(object):
    state_dim_val = 4 * 30
    action_dim_ab = 3
    action_dim_val = action_dim_ab * action_dim_ab
    control_dim_val = state_dim_val // 4 + 3

    def __init__(self, base_model_path, gpu_id=0):
        self.dim = 64
        self.fix_sequence = [
            "black1",
            "black2",
            "black3",
            "black4",
            "boundary",
            "entrance",
            "entrance_sub",
            "white_south",
            "white_north",
            "white_east",
            "white_west",
            "white_m1",
            "white_m2",
            "white_m3",
            "white_m4",
            "garage",
            "room1",
            "room2",
            "living",
            "room3",
            "room4",
            "study_room",
            "kitchen",
            'staircase',
            "bath1",
            "bath2",
            "bath1_sub",
            "storeroom",
            "hallway",
            "dining",
        ]
        self.text_trans = {
            'entrance': '主入口', 'entrance_sub':'次入口', 'hallway': '玄关', 'boundary':'边界',
            'white_south': '南采光', 'white_north': '北采光', 'white_east': '东采光', 'white_west':'西采光',
            'white_m1': '采光1', 'white_m2': '采光2', 'white_m3': '采光3', 'white_m4': '采光4',
            'black1': '黑体1', 'black2': '黑体2', 'black3': '黑体3', 'black4':'黑体4',
            'room1': '卧室1', 'room2': '卧室2', 'room3': '卧室3', 'room4': '卧室4', 'study_room':'书房',
            'living': '客厅', 'dining': '餐厅', 'bath1': '卫生间', 'bath2': '卫生间2', 'bath1_sub': '主卫', 'bath2_sub': '主卫2',
            'kitchen': '厨房', 'staircase': '楼梯',
            'storeroom': '储藏', 'pub': '公共', 'gate':'院门', 'courtyard':'院子', 'porch':'门廊', 'blank':'吹拔','garage':'车库'
        }

        self.room_order = {
            'black1': 0, 'black2': 0, 'black3': 0, 'black4': 0,
            'points_restrain': 1, 'boundary': 2, 'entrance': 3, 'entrance_sub': 4,
            'white_south': 5, 'white_north': 6, 'white_west': 7, 'white_east': 8,

            'white_m1': 9, 'white_m2': 10, 'white_m3': 11, 'white_m4': 12, 'garage': 13,
            'room1': 14, 'room2': 15, 'living': 16, 'room3': 17, 'room4': 18, 'study_room': 19,
            'kitchen': 20, 'staircase': 21,
            'bath1': 22, 'bath2': 23, 'bath1_sub': 24, 'storeroom': 25, 'hallway': 26, 'dining': 27
        }

        self.names_room_down_stair = [
            'room1', 'room2', 'living', 'room3', 'room4', 'study_room', 'garage',
            'kitchen', 'staircase', 'bath1', 'bath2', 'bath1_sub', 'storeroom', 'hallway', 'dining'
        ]

        self.graph_names = [
            'entrance', 'entrance_sub',
            'boundary_west', 'boundary_east', 'boundary_south', 'boundary_north',
            'white_south', 'white_north', 'white_east', 'white_west',
            'black1', 'black2', 'black3', 'black4',
            'white_m1', 'white_m2', 'white_m3', 'white_m4',
            'garage', 'room1', 'room2', 'living', 'room3', 'room4', 'study_room', 'kitchen', 'staircase',
            'bath1', 'bath2', 'bath1_sub', 'storeroom', 'hallway', 'dining'
        ]
        
        self.floor_pointor = 1

        self.state_dim_val = LayoutGenerator.state_dim_val
        self.action_dim_ab = LayoutGenerator.action_dim_ab
        self.action_dim_val = LayoutGenerator.action_dim_val
        self.control_dim_val = LayoutGenerator.control_dim_val

        self.is_graph_purning = False
        self.is_graph_reward = True

        if torch.cuda.is_available() and gpu_id is not None:
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")

        self._load_base_models(base_model_path)

    def _load_base_models(self, base_model_path):
        try:
            self.net_encoder = resnet_encoder().to(self.device)
            path_encoder = os.path.join(base_model_path, "model_encoder.pth")
            self.net_encoder.load_state_dict(torch.load(path_encoder, map_location=self.device))
            self.net_encoder.eval()

            self.net_decoder_a = resnet_decoder_a().to(self.device)
            path_decoder_a = os.path.join(base_model_path, "model_decoder_a.pth")
            self.net_decoder_a.load_state_dict(torch.load(path_decoder_a, map_location=self.device))
            self.net_decoder_a.eval()

            self.net_decoder_b = resnet_decoder_b().to(self.device)
            path_decoder_b = os.path.join(base_model_path, "model_decoder_b.pth")
            self.net_decoder_b.load_state_dict(torch.load(path_decoder_b, map_location=self.device))
            self.net_decoder_b.eval()

            self.reward_model = GAT().to(self.device)
            path_reward_model = os.path.join(base_model_path, "reward_model.pth")
            self.reward_model.load_state_dict(torch.load(path_reward_model, map_location=self.device))
            self.reward_model.eval()

        except Exception as e:
            print(f"error when loading models: {e}")
            raise

    def reset(self, path_in, file, path_graph=False):
        env = [
            "black1",
            "black2",
            "black3",
            "black4",
            "boundary",
            "entrance",
            "entrance_sub",
            "white_south",
            "white_north",
            "white_east",
            "white_west",
        ]

        room = [
            "garage",
            "room1",
            "room2",
            "living",
            "room3",
            "room4",
            "study_room",
            "kitchen",
            "bath1",
            "bath2",
            "bath1_sub",
            "storeroom",
            "hallway",
            "dining",
        ]

        path = path_in + file
        if path.endswith(".xlsx"):
            try:
                df = pd.read_excel(path, index_col=0, sheet_name=f"floor{self.floor_pointor}")
                if self.floor_pointor == 1:
                    excel_file = pd.ExcelFile(path_in + file)
                    sheet_names = excel_file.sheet_names
                    self.floor_count = sum(1 for name in sheet_names if name in ["floor1", "floor2", "floor3"]) 
            except ValueError:
                df = pd.read_excel(path, index_col=0, sheet_name=0)
                if self.floor_pointor == 1:
                    self.floor_count = 1

            if self.is_graph_reward:
                case_graph = Transfer2GraphEdges(layout_ori=df)
                self.graph_controls = case_graph.trans_input_graph_partly(graph_control=False, random=True)
            else:
                self.graph_controls = self.control_parameters()

        elif path.endswith(".csv"):
            df = pd.read_csv(path, index_col=self.floor_pointor - 1)
            self.graph_controls = self.control_parameters()
            self.floor_count = 1
        else:
            raise ValueError("Unsupported file format. Please provide a .xlsx or .csv file.")

        if path_graph is not False:
            self.graph_controls = pd.read_excel(path_graph, sheet_name='graph', index_col=0)

        self.graph_nodes = pd.DataFrame(np.zeros((1, len(self.graph_names))), columns=self.graph_names,  index=['room'])
        for g in self.graph_names:
            if (g in df.columns) or g.startswith('boundary'):
                self.graph_nodes.loc['room', g] = 1

        if self.floor_pointor == 1:
            self.layout_down = None

        self.file_name = file
        if "rural" in self.file_name:
            env = env + ['white_m1', 'white_m2', 'white_m3', 'white_m4']
            if (self.floor_count > 1) and (self.layout_down is not None):
                env.append('staircase')
                df['staircase'] = self.layout_down['staircase']
            else:
                room.append('staircase')
        else:
            room = ['white_m1', 'white_m2', 'white_m3', 'white_m4'] + room
            room.append('staircase')

        self.env = [i for i in self.fix_sequence if i in env]
        self.room = [i for i in self.fix_sequence if i in room]
        self.room_names_all = self.env + self.room

        self.path_in = path_in
        self.env_names = [j for j in self.env if j in df.columns]
        self.room_names = [j for j in self.room if j in df.columns]
        self.names_exist_pure = self.env_names.copy()
        self.fig, self.ax1 = None, None
        self.mask = None
        self.action_available = None
        self.action_available_all = {}
        self.action_choose = None
        self.state_next_available = None
        self.case_pruning = None
        self.df_leaf_points = None

        self.df_origin_data = copy.deepcopy(df[self.env_names + self.room_names])
        self.df_env_full = pd.DataFrame(np.zeros((4, len(self.env))), columns=self.env, index=["x", "y", "w", "d"])
        self.df_env_full[self.env_names] = copy.deepcopy(df[self.env_names])
        self.df_rooms_full = pd.DataFrame(np.zeros((4, len(self.room))), columns=self.room, index=["x", "y", "w", "d"])
        self.df_rooms = pd.DataFrame(index=["x", "y", "w", "d"])

        self.room_pointer = 0

        df_layout = copy.deepcopy(pd.concat((self.df_env_full, self.df_rooms_full), axis=1))
        self.df_layout = df_layout[self.fix_sequence]
        self.df_layout_pure = self.df_layout[self.names_exist_pure]

        self.df_env = copy.deepcopy(df[self.env_names])
        self.df_env = self.df_env.loc[:, self.env_names]
        self.delta_x, self.delta_y = self.compute_delta(self.df_env)
        self.df_env.loc["x", :] += self.delta_x
        self.df_env.loc["y", :] += self.delta_y

        # centralize self.df_rooms_full and self.df_rooms
        self.df_rooms_full.loc["x", :] += self.delta_x
        self.df_rooms_full.loc["y", :] += self.delta_y
        self.df_rooms.loc["x", :] += self.delta_x
        self.df_rooms.loc["y", :] += self.delta_y

        # centralize self.layout_down
        lis_points_wall = None
        if self.layout_down is not None:
            self.layout_down.loc['x', :] += self.delta_x
            self.layout_down.loc['y', :] += self.delta_y
            output = self.get_down_stair_points(df_info_now=self.df_env)
            if output is not None:
                lis_points_wall = output['points_wall']

        rooms_left = [i for i in self.room_names if i not in self.df_rooms]
        matrix_ori = self.trans_input_matrix(
            df_info_now=pd.concat((self.df_env, self.df_rooms), axis=1),
            names_need_all=rooms_left,
            lis_points_online=lis_points_wall
        )
        self.input_matrix = torch.tensor(matrix_ori).to(self.device)

        state_resnet = self.input_matrix.unsqueeze(0).float()
        encode = self.net_encoder.forward(state_resnet)

        # calculate the feasible solutions and possible next states corresponding to the initial state
        self.prepare_available_state(encode=encode)
        state_all = self.prepare_next_state()

        return state_all

    def prepare_available_state(self, encode):
        self.action_available = []
        self.state_next_available = []
        room_name = self.room_names[self.room_pointer]

        # get p1 candidate points
        matrix_p1 = self.net_decoder_a.forward(encode)
        matrix_p1 = torch.sigmoid(matrix_p1)  # get p1 probability matrix
        matrix_mask_p1 = self.mask_leaf_point_p1().to(self.device)
        matrix_p1 = matrix_p1.masked_fill(matrix_mask_p1, value=0)  # p1 pruning
        action_available_a = self.mask_max_point(matrix_p1, self.action_dim_ab)
        if action_available_a is not None:
            for row_id1, col_id1 in action_available_a:
                matrix_p1 = torch.zeros((self.dim, self.dim), dtype=torch.bool).to(self.device)
                matrix_p1[row_id1, col_id1] = True
                matrix_p1 = matrix_p1.unsqueeze(0).unsqueeze(0)
                # get p2 candidate points
                encode_ = torch.concat((encode, matrix_p1), dim=1)
                matrix_p2 = self.net_decoder_b.forward(encode_)
                matrix_p2 = torch.sigmoid(matrix_p2)  # get p2 probability matrix
                matrix_mask_p2 = self.mask_leaf_point_p2(fixed_row=row_id1, fixed_col=col_id1).to(self.device)
                matrix_p2 = matrix_p2.masked_fill(matrix_mask_p2, value=0)  # p2 pruning
                action_available_b = self.mask_max_point(matrix_p2, self.action_dim_ab)
                if action_available_b is not None:
                    for row_id2, col_id2 in action_available_b:
                        matrix_p2 = torch.zeros((self.dim, self.dim), dtype=torch.bool).to(self.device)
                        matrix_p2[row_id2, col_id2] = True
                        matrix_p2 = matrix_p2.unsqueeze(0).unsqueeze(0)
                        # get candidate actions
                        self.action_available.append((matrix_p1, matrix_p2, row_id1, col_id1, row_id2, col_id2))
                        # get candidate next states
                        coord_x_p1, coord_y_p1 = self.matrix2vector_point(row_id=row_id1, col_id=col_id1)
                        coord_x_p2, coord_y_p2 = self.matrix2vector_point(row_id=row_id2, col_id=col_id2)
                        vector = self.transform_room_info(x1=coord_x_p1, y1=coord_y_p1, x2=coord_x_p2, y2=coord_y_p2)
                        df_layout_next = copy.deepcopy(self.df_layout)
                        df_layout_next[room_name] = np.array(vector) - np.array([self.delta_x, self.delta_y, 0, 0])  # update meta info
                        layout_norm = copy.deepcopy(df_layout_next).values
                        layout_parm = layout_norm.T.flatten()
                        self.state_next_available.append(layout_parm)

    def prepare_next_state(self):
        state_available = np.zeros(self.action_dim_val * self.state_dim_val)
        for i in range(len(self.state_next_available)):
            state_available[i * self.state_dim_val : (i + 1) * self.state_dim_val] = self.state_next_available[i]

        self.mask = np.zeros(self.action_dim_val)
        self.mask[: len(self.action_available)] = 1

        controls_array = self.graph_controls.values

        df_node = copy.deepcopy(self.graph_nodes)
        controls_node = df_node.values.T.tolist()

        state_all = {
            'state': state_available,
            'mask': self.mask,
            'control_edges': controls_array,
            'control_nodes': controls_node
        }

        return state_all
    
    def control_parameters(self):
        zeros = np.zeros((len(self.graph_names), len(self.graph_names)))
        df_graph = pd.DataFrame(zeros, columns=self.graph_names, index=self.graph_names)
        return df_graph

    def compute_delta(self, df_env):
        matrix_center_x, matrix_center_y = self.dim * 300 / 2, self.dim * 300 / 2
        w, d = df_env.loc["w", "boundary"], df_env.loc["d", "boundary"]
        boundary_center_x, boundary_center_y = w / 2, d / 2
        delta_x = round((matrix_center_x - boundary_center_x) / 300, 0) * 300
        delta_y = round((matrix_center_y - boundary_center_y) / 300, 0) * 300
        return delta_x, delta_y
    
    def get_down_stair_points(self, df_info_now):
        """
        Args:
            df_info_now: 输入二层房间信息
        Returns:
        """
        df_env_poly = layout2geopandas(layout_info=df_info_now)
        poly_boundary = df_env_poly.loc['rec', 'boundary']
        df_poly_others = copy.deepcopy(df_env_poly).drop('boundary', axis=1)
        poly_env_union = ops.unary_union(df_poly_others.loc['rec', :])
        poly_interior = poly_boundary.difference(poly_env_union)

        width = self.layout_down.loc['w', 'boundary'].item() / 300
        depth = self.layout_down.loc['d', 'boundary'].item() / 300
        x_ori = self.layout_down.loc['x', 'boundary'].item()
        y_ori = self.layout_down.loc['y', 'boundary'].item()

        matrix = [[(x_ori + x * 300, y_ori + y * 300) for x in range(int(width + 1))] for y in range(int(depth + 1))]
        matrix.reverse()

        lis_coord = []
        for p in matrix:
            lis_coord = lis_coord + p
        multi_points = MultiPoint(lis_coord)
        multi_points_interior = multi_points.intersection(poly_interior)

        room_names = [i for i in self.names_room_down_stair if i in self.layout_down]
        lis_points_corner = []
        for r in room_names:
            x, y, w, d = self.layout_down.loc[:, r]
            p1 = [x, y]
            p2 = [x, y + d]
            p3 = [x + w, y]
            p4 = [x + w, y + d]
            for p in [p1, p2, p3, p4]:
                point = Point(p)
                if point.intersects(multi_points_interior) and (p not in lis_points_corner):
                    lis_points_corner.append(p)

        if poly_interior.geom_type == 'MultiPolygon':
            iteration = poly_interior.geoms
        else:
            iteration = [poly_interior]

        for poly in iteration:
            for po in MultiPoint(poly.exterior.coords).geoms:
                if po.intersects(multi_points_interior) and (po not in lis_points_corner):
                    lis_points_corner.append([po.x, po.y])

        outlines = [line.exterior for line in iteration]
        df_poly_rooms = layout2geopandas(layout_info=self.layout_down[room_names])
        for room in df_poly_rooms.columns:
            poly_room = df_poly_rooms.loc['rec', room]
            room_outline = poly_room.exterior
            outlines.append(room_outline)
        union_outline = ops.unary_union(outlines)
        geo_points_online = multi_points_interior.intersection(union_outline)

        if (not geo_points_online.is_empty) and (not multi_points_interior.is_empty):
            lis_points_online = [[i.x, i.y] for i in geo_points_online.geoms]
            lis_points_interior_all = [[i.x, i.y] for i in multi_points_interior.geoms]

            dic = {
                'points_corner': lis_points_corner,
                'points_wall': lis_points_online,
                'points_interior_all': lis_points_interior_all,
                'poly_boundary': poly_boundary,
                'poly_interior': poly_interior
            }
        else:
            return None
        return dic

    def trans_input_matrix(self, df_info_now, names_need_all, lis_points_online=None):
        matrix_input = np.zeros((28, self.dim, self.dim))  # 28 channels
        columns = [i for i in self.room_names_all if i in df_info_now.columns]
        df_info_now = df_info_now[columns]
        for i, name in enumerate(self.room_order.keys()):
            boundary = df_info_now.loc[:, 'boundary'].values.tolist()
            row, col = self.vector_point2matrix(x=boundary[2], y=boundary[3])
            if (row < self.dim) and (col < self.dim):
                if name in df_info_now.columns:
                    x, y, w, d = df_info_now.loc[:, name]
                    start_hor, end_hor, start_ver, end_ver = self.vector_rectangle2matrix(x=x, y=y, w=w, d=d)
                    matrix_input[self.room_order[name], start_ver:end_ver, start_hor:end_hor] = 1
                elif (name not in df_info_now.columns) and (name in names_need_all):
                    matrix_input[self.room_order[name], :, :] = 1

        if lis_points_online is not None:
            for po in lis_points_online:
                row, col = self.vector_point2matrix(x=po[0], y=po[1])
                if (row < self.dim) and (col < self.dim):
                    matrix_input[self.room_order['points_restrain'], row, col] = 1
        else:
            matrix_input[self.room_order['points_restrain'], :, :] = 1

        return matrix_input

    def transform_room_info(self, x1, y1, x2, y2):
        lis_x = [x1, x2]
        lis_y = [y1, y2]
        coord_x = min(lis_x)
        coord_y = min(lis_y)
        width = abs(x1 - x2)
        depth = abs(y1 - y2)
        return coord_x, coord_y, width, depth

    def vector_point2matrix(self, x, y):
        col = int(round(x / 300, 0))
        row = int(self.dim - (round(y / 300, 0)))
        return row, col

    def vector_rectangle2matrix(self, x, y, w, d):
        start_hor = int(round(x / 300, 0))
        end_ver = int(self.dim - (round(y / 300, 0)))
        end_hor = int(start_hor + round(w / 300, 0))
        start_ver = int(end_ver - round(d / 300, 0))
        return start_hor, end_hor, start_ver, end_ver

    def matrix2vector_point(self, row_id, col_id):
        coord_x = col_id * 300
        coord_y = (self.dim - row_id) * 300
        return coord_x, coord_y

    def mask_leaf_point_p1(self):
        case_pruning = get_available_layout(
            new_room_name=self.room_names[self.room_pointer],
            env_info=self.df_env,
            room_info=self.df_rooms,
            control=self.graph_controls if self.is_graph_purning else None,
            layout_down=self.layout_down
        )
        df_points = case_pruning.get_deploy_points()
        self.case_pruning = case_pruning
        self.df_leaf_points = df_points
        matrix_mask = torch.ones((self.dim, self.dim), dtype=torch.bool).to(self.device)
        if isinstance(df_points, pd.DataFrame) and not df_points.empty:
            for col in df_points.columns:
                x, y = df_points.loc["point", col]
                idx, col = self.vector_point2matrix(x, y)
                matrix_mask[idx, col] = False
        return matrix_mask.unsqueeze(0).unsqueeze(0)

    def mask_leaf_point_p2(self, fixed_row, fixed_col):
        matrix_mask = torch.ones((self.dim, self.dim), dtype=torch.bool).to(self.device)
        coord_x, coord_y = self.matrix2vector_point(row_id=fixed_row, col_id=fixed_col)
        for col in self.df_leaf_points.columns:
            x, y = self.df_leaf_points.loc['point', col][0], self.df_leaf_points.loc['point', col][1]
            if x == coord_x and y == coord_y:
                se_point = self.df_leaf_points[col]
                poly_rooms_potential = self.case_pruning.get_available_layout_fixed_point(se_point=se_point)

                for poly in poly_rooms_potential:
                    x_potential = [poly[0], poly[0] + poly[2]]
                    y_potential = [poly[1], poly[1] + poly[3]]
                    x_available = None
                    y_available = None
                    for x_p in x_potential:
                        if x_p != x:
                            x_available = x_p
                    for y_p in y_potential:
                        if y_p != y:
                            y_available = y_p
                    idx, col = self.vector_point2matrix(x_available, y_available)
                    matrix_mask[idx, col] = False

        return matrix_mask.unsqueeze(0).unsqueeze(0)

    def mask_max_point(self, matrix, rank):
        action_available = []
        rows, cols = matrix.size()[-2:]
        flattened_matrix = matrix.view(-1)
        topk_values, topk_indices = torch.topk(flattened_matrix, rank)
        for i in range(rank):
            if topk_values[i] < 1e-5:
                break
            else:
                row_index, col_index = divmod(topk_indices[i].item(), cols)
                action_available.append((row_index, col_index))

        return action_available

    def action_trans_coord(self, action):
        x, y, w, d = action.values
        if w < 0:
            x = x + w
        if d < 0:
            y = y + d
        return [x, y, abs(w), abs(d)]

    def step(self, action, value_out, log_dir, is_visible):
        start_time = time.time()
        room_name = self.room_names[self.room_pointer]
        self.action_available_all[room_name] = [self.action_available, value_out]

        _, _, row_id1, col_id1, row_id2, col_id2 = self.action_available[action]
        coord_x_p1, coord_y_p1 = self.matrix2vector_point(row_id=row_id1, col_id=col_id1)
        coord_x_p2, coord_y_p2 = self.matrix2vector_point(row_id=row_id2, col_id=col_id2)
        vector_pre = self.transform_room_info(x1=coord_x_p1, y1=coord_y_p1, x2=coord_x_p2, y2=coord_y_p2)

        post_process = PostProcess(
            self.df_layout_pure,
            self.graph_controls if self.is_graph_purning else None,
            room_name,
            vector_pre,
            self.delta_x,
            self.delta_y
        )
        matrix_out, vector = post_process.fix_gap(need_fix=True)
        self.action_choose = pd.Series(
            np.array(vector) - np.array([self.delta_x, self.delta_y, 0, 0]), index=["x", "y", "w", "d"]
        )
        self.names_exist_pure.append(room_name)
        channel_id = self.room_order[room_name]
        self.input_matrix[channel_id] = torch.tensor(matrix_out).to(self.device)
        self.df_rooms_full[room_name] = vector
        self.df_rooms[room_name] = vector
        self.df_layout[room_name] = np.array(vector) - np.array([self.delta_x, self.delta_y, 0, 0])
        self.df_layout_pure = self.df_layout[self.names_exist_pure]

        self.room_pointer += 1

        # terminated、truncated、reward、next_state
        if self.room_pointer >= len(self.room_names):
            terminated = True
            truncated = False
        else:
            state_resnet = self.input_matrix.unsqueeze(0).float()
            encode = self.net_encoder.forward(state_resnet)
            self.prepare_available_state(encode=encode)
            terminated = False
            truncated = len(self.action_available) == 0
        trans_time = time.time()

        if terminated or truncated:
            # calculate the artificially designed part of the sparse reward
            total_score, dic_score = layout_evaluator(
                df_info=copy.deepcopy(self.df_layout_pure),
                room_names_all=self.room_names,
                mode="train",
                graph=self.graph_controls if self.is_graph_reward else None,
            )
            reward = total_score
            if np.isnan(reward) or np.isinf(reward):
                print(f"{self.file_name}:Reward_o is {reward}")
                print(self.df_layout_pure)
            # calculate the model reward part of the sparse reward
            case = Transfer2Graph(layout_ori=self.df_layout_pure)
            data = case.markov_data_trans()
            data_batch = Batch.from_data_list([data]).to(self.device)
            reward += 1.0 * self.reward_model.forward(data_batch).item()
            if np.isnan(reward) or np.isinf(reward):
                print(f"{self.file_name}:Reward_n is {reward}")
                print(self.df_layout_pure)
            # finalize and log the layout
            self.close(log_dir, is_visible)
            self.floor_pointor += 1
        else:
            reward = 0
        reward_time = time.time()

        if (terminated or truncated) and (self.floor_pointor < (self.floor_count + 1)):
            self.layout_down = self.df_layout[self.env_names + self.room_names]
            state_all = self.reset(path_in=self.path_in, file=self.file_name)
        else:
            state_all = self.prepare_next_state()
        finish_time = time.time()

        if (finish_time - start_time) > 5:
            print(
                f"Case {self.file_name} timeout {(finish_time-start_time)}s. trans:{(trans_time-start_time):.2f}s, reward:{reward_time-trans_time:.2f}s, prepare:{(finish_time-reward_time)}s."
            )
        return state_all, reward, terminated, truncated, {}
    
    def wrap_text_mixed(self, s, max_width):
        lines = []
        current = ""
        for ch in s:
            if wcswidth(current + ch) > max_width:
                lines.append(current)
                current = ch
            else:
                current += ch
        if current:
            lines.append(current)
        return lines
    
    def adjust_text_and_rect(self, ax, texts, rect_width, line_height):
        labels, values = texts

        num_lines = sum(value.count("\n") + 1 for value in values)
        rect_height = (num_lines + 0.5) * line_height
        rect_x = 1 - rect_width
        rect_y = 1 - rect_height
        rect = Rectangle(
            (rect_x, rect_y), rect_width, rect_height,
            linewidth=2, edgecolor="black", facecolor="white", transform=ax.transAxes
        )
        ax.add_patch(rect)
        rect.set_zorder(6)

        current_line = 0
        for label, value in zip(labels, values):
            value_lines = value.split("\n")
            for i, line in enumerate(value_lines):
                text_y = 1 - (current_line + 1) * line_height
                if i == 0:
                    ax.text(
                        rect_x + 0.02, text_y, label,
                        transform=ax.transAxes, ha="left", fontsize=12, zorder=7
                    )
                ax.text(
                    rect_x + 0.16, text_y, line,
                    transform=ax.transAxes, ha="left", fontsize=12, zorder=7
                )
                current_line += 1

    def render(self):
        if  self.action_choose is not None:
            df_shapes = layout2geopandas(layout_info=self.df_layout_pure)
        else:
            df_shapes = layout2geopandas(layout_info=self.df_origin_data)
        self.render_df(df_shapes)

    def render_df(self, df_shapes):
        if self.fig is None or self.ax1 is None:
            self.fig = plt.figure(figsize=(8, 8))
            gs = self.fig.add_gridspec(1, 1)
            self.ax1 = self.fig.add_subplot(gs[0, 0])
        else:
            self.ax1.cla()
        
        case_draw = GraphPlan(layout_info=df_shapes, file_name=self.file_name, path_out="", if_save=False)
        case_draw.draw_plan(self.action_choose, ax=self.ax1)

        if self.action_choose is not None:
            room_name = self.room_names[self.room_pointer - 1]
            if room_name in self.action_available_all:
                action_list, _ = self.action_available_all[room_name]
                for action_available in action_list:
                    _, _, row_id1, col_id1, row_id2, col_id2 = action_available
                    coord_x_p1, coord_y_p1 = self.matrix2vector_point(row_id=row_id1, col_id=col_id1)
                    coord_x_p2, coord_y_p2 = self.matrix2vector_point(row_id=row_id2, col_id=col_id2)
                    vector = self.transform_room_info(x1=coord_x_p1, y1=coord_y_p1, x2=coord_x_p2, y2=coord_y_p2)
                    action_available_i = pd.Series(
                        np.array(vector) - np.array([self.delta_x, self.delta_y, 0, 0]), index=["x", "y", "w", "d"]
                    )
                    x_, y_, w_, d_ = action_available_i.values
                    rect = Rectangle((x_, y_), w_, d_, linewidth=2, edgecolor="grey", facecolor="none", linestyle="--")
                    self.ax1.add_patch(rect)
                    rect.set_zorder(6)

        labels = ["File_name:", "Floor:", "Total_num:", "Current_num:"]
        wrapped = self.wrap_text_mixed(self.file_name, max_width=31)
        file_name_wrapped = "\n".join(wrapped)
        values = [file_name_wrapped, str(self.floor_pointor), str(len(self.room_names)), str(self.room_pointer)]
        texts = (labels, values)
        self.adjust_text_and_rect(
            ax=self.ax1,
            texts=texts,
            rect_width=0.5,
            line_height=0.03
        )

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

    def save_file(self, save_path):
        try:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_name_without_extension = os.path.splitext(self.file_name)[0]
            full_save_path = os.path.join(save_path, file_name_without_extension + ".xlsx")
            os.makedirs(os.path.dirname(full_save_path), exist_ok=True)

            with pd.ExcelWriter(full_save_path, engine="openpyxl") as writer:
                self.df_layout_origin = self.df_layout_pure
                self.df_layout_origin.to_excel(writer, sheet_name=f"floor{self.floor_pointor}", index=True)
                self.graph_controls.to_excel(writer, sheet_name=f"floor{self.floor_pointor}_graph", index=True)
                if self.layout_down is not None:
                    layout_down = copy.deepcopy(self.layout_down)
                    layout_down.loc['x', :] - self.delta_x
                    layout_down.loc['y', :] - self.delta_y
                    layout_down.to_excel(writer, sheet_name=f"floor{self.floor_pointor-1}", index=True)
                
                for room_name, (action_list, value_out) in self.action_available_all.items():
                    action_data = []
                    for i, action_available in enumerate(action_list):
                        _, _, row_id1, col_id1, row_id2, col_id2 = action_available
                        coord_x_p1, coord_y_p1 = self.matrix2vector_point(row_id=row_id1, col_id=col_id1)
                        coord_x_p2, coord_y_p2 = self.matrix2vector_point(row_id=row_id2, col_id=col_id2)
                        vector = self.transform_room_info(x1=coord_x_p1, y1=coord_y_p1, x2=coord_x_p2, y2=coord_y_p2)
                        x, y, w, d = np.array(vector) - np.array([self.delta_x, self.delta_y, 0, 0])
                        score = value_out[i]
                        action_data.append([x, y, w, d, score])

                    action_df = pd.DataFrame(action_data, columns=["x", "y", "w", "d", "score"]).T
                    action_df.to_excel(writer, sheet_name=f"floor{self.floor_pointor}_{room_name}", index=True)

        except Exception as e:
            print("save_file:" + str(e))
            print(traceback.format_exc())

    def save_pic(self, save_path):
        try:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            file_name_without_extension = os.path.splitext(self.file_name)[0]
            full_save_path = os.path.join(save_path, file_name_without_extension + f"_floor{self.floor_pointor}" + ".jpg")
            os.makedirs(os.path.dirname(full_save_path), exist_ok=True)

            plt.savefig(full_save_path, dpi=300)
            plt.close(self.fig)
        except Exception as e:
            print("save_pic:" + str(e))
            print(traceback.format_exc())

    def close(self, log_dir, is_visible):
        save_path = log_dir + "/results/"
        if is_visible:
            self.render()
            self.save_pic(save_path)
            self.save_file(save_path)

    def error_log(self, log_dir):
        save_path = log_dir + "/error_log/"
        self.render()
        self.save_pic(save_path)
        self.save_file(save_path)

    def prepare_prompt(self):
        """
        Generate environment image and prompt messages for QwenVL inference.
        This method requires QwenVL utilities to be installed.
        """
        if not QWENVL_UTILS_AVAILABLE:
            raise ImportError(
                "QwenVL utilities are not installed. "
                "Please install them using: pip install -r requirements_qwen.txt"
            )

        file_name_without_extension = os.path.splitext(self.file_name)[0]

        # print(f"Generating env image from layout for {file_name_without_extension}...")
        df_env = copy.deepcopy(self.df_env)
        df_env.loc["x", :] -= self.delta_x
        df_env.loc["y", :] -= self.delta_y
        for col in df_env.columns:
            if col in self.text_trans.keys():
                df_env.rename(columns={col: self.text_trans[col]}, inplace=True)

        from layout_visualization import GraphOriginal
        case = GraphOriginal(layout_info=df_env, file_name=file_name_without_extension, path_out=self.path_in)
        case.draw_plan()

        image_path = os.path.join(self.path_in, f"{file_name_without_extension}.jpeg")
        resized_image_path = image_path.replace(".jpeg", "_resized.jpeg")
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        image.save(resized_image_path)
        # print(f"Resized image saved to {resized_image_path}")

        room_lis = []
        for room in self.room_names:
            room_lis.append(self.text_trans[room])

        # print("Generating prompt...")
        messages = generate_prompt_from_file(df_env, room_lis, file_name_without_extension, resized_image_path)
        return messages

    def save_llm_result(self, llm_result, save_path):
        """
        Save LLM inference results to Excel file.
        This method requires QwenVL utilities to be installed.
        """
        if not QWENVL_UTILS_AVAILABLE:
            raise ImportError(
                "QwenVL utilities are not installed. "
                "Please install them using: pip install -r requirements_qwen.txt"
            )

        try:
            # Create save directory
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Create reverse mapping from Chinese to English
            reverse_trans = {v: k for k, v in self.text_trans.items()}

            # Convert JSON string to dictionary
            result_dict = json.loads(llm_result)

            # Convert room names and create new dictionary
            converted_dict = {}
            for zh_name, coords in result_dict.items():
                if zh_name in reverse_trans:
                    eng_name = reverse_trans[zh_name]
                    converted_dict[eng_name] = coords

            # Create DataFrame
            df_llm = pd.DataFrame(
                {name: coords for name, coords in converted_dict.items()},
                index=['x', 'y', 'w', 'd']
            )

            # Merge LLM results with environment data
            df_env = copy.deepcopy(self.df_env)
            df_env.loc["x", :] -= self.delta_x
            df_env.loc["y", :] -= self.delta_y
            df_combined = pd.concat([df_env, df_llm], axis=1)

            # Build save path
            file_name_without_extension = os.path.splitext(self.file_name)[0]
            full_save_path = os.path.join(save_path, f"{file_name_without_extension}_llm.xlsx")

            # Save as Excel file
            with pd.ExcelWriter(full_save_path, engine="openpyxl") as writer:
                # Save merged data to floor sheet
                df_combined.to_excel(writer, sheet_name=f"floor{self.floor_pointor}", index=True)

                # Save original LLM output to llm_output sheet
                df_llm.to_excel(writer, sheet_name=f"floor{self.floor_pointor}_llm", index=True)

            # print(f"LLM result saved to {full_save_path}")

            df_shapes = layout2geopandas(layout_info=df_combined)
            self.render_df(df_shapes)
            self.save_pic(save_path)

        except Exception as e:
            print(f"Error in saving LLM result: {str(e)}")
            print(traceback.format_exc())

