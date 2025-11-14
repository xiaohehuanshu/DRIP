import numpy as np
import pandas as pd
from shapely import Point, LineString, ops
from tools_layout_modeling import build_room, layout2geopandas
from pruning.get_leaf_node_edges import get_available_layout


class PostProcess:
    def __init__(self, df_info: pd.DataFrame, control, room_name, vector, delta_x, delta_y):
        self.super_parm = 600
        self.dim = 64
        self.room_name = room_name
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.vector = vector
        self.control = control
        self.df_info = df_info.copy()
        self.df_info.loc["x", :] = df_info.loc["x", :] + self.delta_x
        self.df_info.loc["y", :] = df_info.loc["y", :] + self.delta_y
        self.gdf_layout = layout2geopandas(layout_info=self.df_info)
        self.boundary = self.gdf_layout.loc["rec", "boundary"]
        self.blank_edges = self.get_blank_edges()

        self.names_env = [
            "black1",
            "black2",
            "black3",
            "black4",
            "boundary",
            "entrance",
            "white_south",
            "white_north",
            "white_third",
            "white_fourth",
            "white_east",
            "white_west",
        ]

        names_env_sub = [i for i in self.df_info.columns if i in self.names_env]
        names_room_sub = [i for i in self.df_info.columns if i not in self.names_env]
        df_env = self.df_info.loc[:, names_env_sub]
        df_room = self.df_info.loc[:, names_room_sub]
        self.case_pruning = get_available_layout(
            new_room_name=self.room_name, env_info=df_env, room_info=df_room, control=self.control
        )

    def fix_gap(self, need_fix):
        if need_fix:
            edges, info, _ = self.get_new_room_edges()
            x, y, w, d = info

            for idx, r in enumerate(edges):
                for b in self.blank_edges:
                    judge = self.judge_parallel_overlap(source_line=r, target_line=b)
                    if judge:
                        dis = r.distance(b)
                        if (dis < self.super_parm) and (dis > 0):
                            if idx == 0:
                                mid_point_b_x = b.centroid.x
                                w = w - (mid_point_b_x - x)
                                x = mid_point_b_x
                            elif idx == 1:
                                mid_point_b_y = b.centroid.y
                                d = d + (mid_point_b_y - (y + d))
                            elif idx == 2:
                                mid_point_b_x = b.centroid.x
                                w = w + (mid_point_b_x - (x + w))
                            else:
                                mid_point_b_y = b.centroid.y
                                d = d - (mid_point_b_y - y)
                                y = mid_point_b_y
            vector_new_center = [x, y, w, d]
            hor0, hor1, ver0, ver1 = self.vector_rectangle2matrix(
                x=vector_new_center[0], y=vector_new_center[1], w=vector_new_center[2], d=vector_new_center[3]
            )

            room_poly = build_room(x, y, w, d)

            judge1 = self.judge_room_in_law(poly=room_poly)
            judge2 = self.case_pruning.if_violation(room=room_poly)

            if (not judge1) and (not judge2):
                matrix = np.zeros((self.dim, self.dim))
                matrix[ver0:ver1, hor0:hor1] = True
                return matrix, vector_new_center

        hor0, hor1, ver0, ver1 = self.vector_rectangle2matrix(
            x=self.vector[0], y=self.vector[1], w=self.vector[2], d=self.vector[3]
        )
        matrix = np.zeros((self.dim, self.dim))
        matrix[ver0:ver1, hor0:hor1] = True
        return matrix, self.vector

    def judge_room_in_law(self, poly):
        if poly.area < 1e-9:
            return True
        else:
            return False

    def make_line_room(self):
        for col in self.gdf_layout.columns:
            room_poly = self.gdf_layout.loc["rec", col]
            room_line = room_poly.exterior
            self.gdf_layout.loc["line", col] = room_line

    def get_blank(self):
        without_boundary = self.gdf_layout.drop(columns=["boundary"]).copy()
        shapes_union = ops.unary_union(without_boundary.loc["poly"].values)
        poly_blank = self.boundary.difference(shapes_union)
        return poly_blank

    def project_line_to_line(self, source_line: LineString, target_line: LineString) -> LineString:
        target_coords = list(target_line.coords)
        A = np.array(target_coords[0])
        B = np.array(target_coords[-1])
        AB = B - A

        if np.linalg.norm(AB) < 1e-9:
            raise ValueError("Target line has zero length.")

        def project_point(P):
            AP = P - A
            t = np.dot(AP, AB) / np.dot(AB, AB)
            return A + t * AB

        proj_points = []
        for p in list(source_line.coords):
            P = np.array(p)
            proj = project_point(P)
            proj_points.append((proj[0], proj[1]))

        return LineString(proj_points)

    def judge_parallel_overlap(self, source_line: LineString, target_line: LineString) -> bool:
        project_line = self.project_line_to_line(source_line, target_line)
        if project_line.intersects(target_line) and (project_line.length == source_line.length):
            return True
        else:
            return False

    def get_new_room_edges(self):
        edges = []
        x, y, w, d = self.vector
        corner1 = Point([x, y])
        corner2 = Point([x, y + d])
        corner3 = Point([x + w, y + d])
        corner4 = Point([x + w, y])
        lis_corners = [corner1, corner2, corner3, corner4]
        for i in range(len(lis_corners)):
            point_start = lis_corners[i]
            point_end = lis_corners[(i + 1) % len(lis_corners)]
            edge = ops.LineString([point_start, point_end])
            edges.append(edge)
        return edges, [x, y, w, d], [corner1, corner2, corner3, corner4]

    def get_blank_edges(self):
        poly_blank = self.get_blank()
        if poly_blank.geom_type == "Polygon":
            iteration_blank = [poly_blank]
        else:
            iteration_pure_new = poly_blank.geoms
            iteration_blank = [i for i in iteration_pure_new]

        edges_blank = []
        for blank in iteration_blank:
            coords_blank = list(blank.exterior.coords)
            points_blank = coords_blank[:-1]

            for p in range(len(points_blank)):
                start_blank = points_blank[p]
                end_blank = points_blank[(p + 1) % len(points_blank)]
                edges_blank.append(LineString([start_blank, end_blank]))
        return edges_blank

    def vector_rectangle2matrix(self, x, y, w, d):
        start_hor = int(round(x / 300, 0))
        end_ver = int(self.dim - (round(y / 300, 0)))
        end_hor = int(start_hor + round(w / 300, 0))
        start_ver = int(end_ver - round(d / 300, 0))
        return start_hor, end_hor, start_ver, end_ver
