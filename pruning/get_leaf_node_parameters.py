import numpy as np
import pandas as pd
from shapely import geometry
from threading import local

data = local()

def design_parameters(dir_room_relation, dir_room_location, dir_env_info, dir_size_bounds, dir_size_bounds_win):
    dic = {
        'room_relation': dir_room_relation,
        'room_location': dir_room_location,
        'env_info': dir_env_info,
        'size_bounds': dir_size_bounds,
        'size_bounds_win': dir_size_bounds_win
    }
    return dic

parm = design_parameters(dir_room_relation='./pruning/criterion_leaf_node_pruning/Room_Relation.xlsx',
                         dir_room_location='./pruning/criterion_leaf_node_pruning/Room_Location.xlsx',
                         dir_env_info='case3-1.xlsx',
                         dir_size_bounds='./pruning/criterion_leaf_node_pruning/Size_Bounds_loose.xlsx',
                         dir_size_bounds_win='./pruning/criterion_leaf_node_pruning/Size_bounds_windows.xlsx')


def room_relation(path=parm['room_relation']):
    if not hasattr(data, 'room_adjacent_matrix'):
        data.room_adjacent_matrix = pd.read_excel(path, index_col=0)
    return data.room_adjacent_matrix


def room_location(path=parm['room_location']):
    if not hasattr(data, 'room_location_info'):
        data.room_location_info = pd.read_excel(path, index_col=0)
    room_location_info = data.room_location_info
    if not hasattr(data, 'room_location_points'):
        room_location_points = pd.DataFrame(
            np.zeros(len(room_location_info.columns)).reshape(1, len(room_location_info.columns)),
            columns=room_location_info.columns)
        columns = []
        points = []
        # print(room_location_info)
        for i in room_location_info.columns:
            if room_location_info[i][0] == -1 or room_location_info[i][1] == -1:
                continue
            columns.append(i)
            point = geometry.Point(room_location_info[i][0], room_location_info[i][1])
            points.append(point)
            # room_location_points[i] = point
        room_location_points = pd.DataFrame([points], columns=columns)
        data.room_location_points = room_location_points

    # print('room_location_points')
    # print(data_test.room_location_points)
    return data.room_location_points


def environment_info(path=parm['env_info']):
    if not hasattr(data, 'env_info'):
        data.env_info = pd.read_excel(path, index_col=0)

    # print('env_info')
    # print(data_test.env_info)
    return data.env_info


def size_limits(path=parm['size_bounds']):
    global data
    if not hasattr(data, "size_bounds"):
        data.size_bounds = pd.read_excel(path, index_col=0)
    # print('size_bounds')
    # print(data_test.size_bounds)
    return data.size_bounds


def size_limits_windows(path=parm['size_bounds_win']):
    if not hasattr(data, 'size_bounds_win'):
        data_win = pd.read_excel(path, index_col=0)
        data_win_se = data_win.loc['width_min', :]
        data.size_bounds_win = data_win_se
    return data.size_bounds_win

# def room_limits(name):
#     global data_test
#     if not hasattr(data_test, "room_info_total"):
#         sizes = size_limits()
#         room_info_total = {}
#         for i in sizes.columns:
#             room_info_total[i] = RoomLimit(i, sizes[i].values)
#         # room_info_total = {
#         #     'white_m1': RoomLimit('white_m1', sizes['white_m1'].values),
#         #     'white_m2': RoomLimit('white_m2', sizes['white_m2'].values),
#         #     'room1': RoomLimit('room1', sizes['room1'].values),
#         #     'room2': RoomLimit('room2', sizes['room2'].values),
#         #     'living': RoomLimit('living', sizes['living'].values),
#         #     'dinner': RoomLimit('dinner', sizes['dinner'].values),
#         #     'kitchen': RoomLimit('kitchen', sizes['kitchen'].values),
#         #     'toilet_main': RoomLimit('toilet_main', sizes['toilet_main'].values)
#         # }
#         data_test.room_info_total = room_info_total
#     if name in data_test.room_info_total.keys():
#         #
#         # print('room_info_total')
#         # print(data_test.room_info_total.keys())
#         return data_test.room_info_total[name]
#     else:
#         print(False)
#         raise Exception('Invalid room_name')


# class RoomLimit:
#     # def __init__(self, room_name, w_min, w_max, d_min, d_max, area_min, area_max):
#     def __init__(self, room_name, values):
#         self.room_name = room_name

#         self.w_min = values[0]
#         self.w_max = values[1]
#         self.d_min = values[2]
#         self.d_max = values[3]

#         self.area_min = values[4]
#         self.area_max = values[5]
#         self.searching_grade = 300
#         self.available_sizes = self.available_size()
#
#     def available_size(self):
#         w_list = np.arange(self.w_min, self.w_max, self.searching_grade)
#         if w_list[-1] < self.w_max:
#             w_list = np.append(w_list, self.w_max)
#
#         d_list = np.arange(self.d_min, self.d_max, self.searching_grade)
#         if d_list[-1] < self.d_max:
#             d_list = np.append(d_list, self.d_max)
#         available_wd = []
#         for i in w_list:
#             for j in d_list:

#                 area = i * j / 1000000
#                 if self.area_min <= area <= self.area_max:
#                     available_wd.append((i, j))
#         return available_wd


class RoomSizeLimits:
    def __init__(self):
        self.searching_grade = 300

    def available_sizes_all(self):
        sizes_total = {}
        sizes = size_limits()
        for i in sizes.columns:
            values_ = sizes[i].values
            sizes_total[i] = self.available_size(values=values_)
        return sizes_total

    def area_limits_all(self):
        area_limits_all = {}
        sizes = size_limits()
        for i in sizes.columns:
            values_ = sizes[i].values
            area_min = values_[4]
            area_max = values_[5]
            area_limits_all[i] = [area_min, area_max]
        return area_limits_all

    def window_limits(self):
        return size_limits_windows()

    def available_size(self, values):
        w_min = values[0]
        w_max = values[1]
        d_min = values[2]
        d_max = values[3]
        area_min = values[4]
        area_max = values[5]
        w_list = np.arange(w_min, w_max, self.searching_grade)
        if w_list[-1] < w_max:
            w_list = np.append(w_list, w_max)

        d_list = np.arange(d_min, d_max, self.searching_grade)
        if d_list[-1] < d_max:
            d_list = np.append(d_list, d_max)
        available_wd = []
        for i in w_list:
            for j in d_list:
                area = i * j / 1000000
                if area_min <= area <= area_max:
                    available_wd.append((i, j))

        return available_wd


room_limits = RoomSizeLimits()
room_sizes = room_limits.available_sizes_all()
area_limits = room_limits.area_limits_all()
window_limits = room_limits.window_limits()


# if __name__ == '__main__':
#     from graph import location_graph

#     env_information = environment_info()
#     location_graph(env_information)
#     # storage?
#     room_lists = ['white_m1', 'white_m2', 'room1', 'room2', 'living', 'kitchen', 'toilet_main', 'dinner', 'storage']
#     for room_name in room_lists:
#         available_size = room_limits(room_name).available_size()
#         print(room_name)
#         print(len(available_size))
