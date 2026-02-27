"""
graphs that depend on the original excel files without transformation
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pylab as mpl
import openpyxl
import os

class GraphOriginal:
    def __init__(self, layout_info, file_name, path_out):
        self.layout_info = layout_info
        self.path_out = path_out
        self.file_name = file_name
        self.text_size = 18
        self.text_size_sub = 14

        # Normalize dataframe format
        self.layout_info.index = ['x', 'y', 'w', 'd']
        wrong_names = ['dinning', 'dinner', 'toilet_main', 'toilet2', 'toilet3', 'toilet4']
        wrong_name_trans = {
            'dinning': 'dining', 'dinner': 'dining',
            'toilet_main': 'bath1', 'toilet2': 'bath2', 'toilet3': 'bath3', 'toilet4': 'bath4'
        }
        for i in self.layout_info.columns:
            if i in wrong_names:
                self.layout_info = self.layout_info.rename(columns={i: wrong_name_trans[i]})

        # Adjust room order
        self.name_list = [
            'boundary', 'entrance', 'white_south', 'white_north', 'white_third', 'white_fourth',
            'black1', 'black2', 'black3', 'black4',
            'white_m1', 'white_m2', 'room1', 'room2', 'living', 'room3', 'room4',
            'kitchen', 'bath1', 'bath1_sub', 'dining', 'storeroom'
        ]
        name_list = [i for i in self.name_list if i in self.layout_info.columns]
        self.layout_info = self.layout_info[name_list]

        # Floor plan layout
        self.graph_color = {
            'boundary': '#FFE27D', 'entrance': 'whitesmoke', 'hallway': '#FFE27D',
            'white_south': 'whitesmoke', 'white_north': 'whitesmoke', 'white_east': 'whitesmoke', 'white_west':'whitesmoke',
            'white_m1': 'whitesmoke', 'white_m2': 'whitesmoke', 'white_m3': 'whitesmoke', 'white_m4': 'whitesmoke',
            'black1': 'darkgrey', 'black2': 'darkgrey', 'black3': 'darkgrey', 'black4': 'darkgrey',
            'room1': '#F3A169', 'room2': '#F3A169', 'room3': '#F3A169', 'room4': 'lightgray',
            'living': '#FFE27D', 'dining': '#C5D5AE',
            'bath1': '#6483A6', 'bath2': '#90AEC4', 'bath3': '#90AEC4', 'bath1_sub': '#90AEC4', 'bath2_sub': '#90AEC4',
            'kitchen': '#80A6AF', 'storage':'#90AEC4', 'storeroom':'#90AEC4',
            'gate':'whitesmoke', 'courtyard':'whitesmoke', 'staircase':'#8474A0', 'porch':'whitesmoke', 'study_room':'#F3A169',
            'blank':'whitesmoke', 'entrance_sub':'whitesmoke', 'garage':'gray'
        }

        self.text_trans = {
            'entrance': 'Entrance', 'entrance_sub':'Entrance_sub', 'hallway': 'hallway',
            'white_south': 'Light', 'white_north': 'Light', 'white_east': 'Light', 'white_west':'Light',
            'white_m1': 'Light1', 'white_m2': 'Light2', 'white_m3': 'Light3', 'white_m4': 'Light4',
            'black1': '', 'black2': '', 'black3': '', 'black4':'',
            'room1': 'Room1', 'room2': 'Room2', 'room3': 'Room3', 'room4': 'Room4', 'study_room':'Study',
            'living': 'Living', 'dining': 'Dining',
            'bath1': 'Bath1', 'bath2': 'Bath2', 'bath3': 'Bath3', 'bath1_sub': 'Bath1_sub', 'bath2_sub': 'Bath2_sub',
            'kitchen': 'Kitchen', 'staircase': 'Stair', 'storeroom': 'Storeroom',
            'storage': 'Storage', 'pub': '', 'gate':'Gate', 'courtyard':'Courtyard', 'porch':'Porch',
            'blank':'Blank', 'garage':'Garage'
        }

    def draw_plan(self):
        # mpl.rcParams['font.sans-serif'] = ['simsun']  # Specify default font
        mpl.rcParams['font.sans-serif'] = ['Times New Roman']  # Specify default font
        mpl.rcParams['axes.unicode_minus'] = False  # Fix the issue where minus sign '-' displays as a square when saving images

        fig = plt.figure('layout', figsize=(9, 9))
        ax1 = plt.subplot(1, 1, 1)
        plt.ion()  # Enable interactive mode

        name_list = self.layout_info.columns.values.tolist()
        # print(name_list)
        # name_list.reverse()
        # name_list = ['boundary']+name_list
        # lis_exclude = ['living', 'dining']
        lis_exclude = []

        if 'entrance' in name_list:
            name_list.remove('entrance')
            name_list = name_list + ['entrance']

        for i, item in enumerate(name_list):
            tmp = self.layout_info[item]
            X, Y, W, H = tmp.values
            if item not in lis_exclude:
                rect = plt.Rectangle((X, Y), W, H, fill=True, facecolor=self.graph_color[item], alpha=0.8, lw=3)
                ax1.add_patch(rect)
                rect1 = plt.Rectangle((X, Y), W, H, fill=False, edgecolor='black', alpha=0.9, lw=3)
                ax1.add_patch(rect1)
                rect.set_label(item)

            # if item not in ['white_m1', 'whtie_m2', 'white_m3']:
            if item != 'boundary':
                plt.text(
                    X + W / 2, Y + H / 2,
                    s='%s' % (self.text_trans[item]),
                    color='black',
                    verticalalignment='center',
                    horizontalalignment='center',
                    size=self.text_size
                    )
        # Draw auxiliary dashed lines
        # for i, item in enumerate(name_list):
        #     if item not in ['living', 'dinner']:
        #         tmp = self.layout_info[item]
        #         X, Y, W, H = tmp.values
        #         rect1 = plt.Rectangle((X, Y), W, H, fill=False, ls='--', edgecolor='black', alpha=0.5, lw=3)
        #         ax1.add_patch(rect1)

        plt.xlim([-3000, 18000])
        plt.ylim([-3000, 18000])
        # plt.title(self.file_name, fontdict={'fontsize': self.text_size_sub})
        # plt.xticks(fontproperties='Times New Roman', size=self.text_size_sub)
        # plt.yticks(fontproperties='Times New Roman', size=self.text_size_sub)
        plt.pause(1)
        plt.tight_layout(pad=0.05)
        fig.subplots_adjust(wspace=0.2, hspace=0.2)
        # plt.show()
        plt.savefig(self.path_out + str(self.file_name) + '.jpeg', dpi=300)
        plt.close(fig=fig)

    def draw_environment(self):
        env_names = ['boundary', 'entrance', 'black1', 'white_south', 'white_north', 'white_third']

        plt.figure('layout', figsize=(9, 9))
        ax1 = plt.subplot(1, 1, 1)
        # plt.cla()

        name_list = self.layout_info.columns.values.tolist()[1:]
        # name_list.reverse()
        name_list = ['boundary']+name_list
        if 'entrance' in name_list:
            name_list.remove('entrance')
            name_list = name_list + ['entrance']

        for i, item in enumerate(name_list):
            if item in env_names:
                tmp = self.layout_info[item]
                X, Y, W, H = tmp.values
                if item == 'boundary':
                    rect = plt.Rectangle((X, Y), W, H, fill=True, facecolor='gainsboro', alpha=1, lw=3)
                    ax1.add_patch(rect)
                    rect1 = plt.Rectangle((X, Y), W, H, fill=False, edgecolor='black', alpha=0.9, lw=3)
                    ax1.add_patch(rect1)
                    rect.set_label(item)
                else:
                    rect = plt.Rectangle((X, Y), W, H, fill=True, facecolor=self.graph_color[item], alpha=1, lw=3)
                    ax1.add_patch(rect)
                    rect1 = plt.Rectangle((X, Y), W, H, fill=False, edgecolor='black', alpha=0.9, lw=3)
                    ax1.add_patch(rect1)
                    rect.set_label(item)

                    plt.text(
                        X + W / 2, Y + H / 2,
                        s='%s' % (self.text_trans[item]),
                        color='black',
                        verticalalignment='center',
                        horizontalalignment='center',
                        size=self.text_size
                    )

        plt.xlim([-3000, 15000])
        plt.ylim([-3000, 15000])
        # plt.title(self.file_name, fontdict={'fontsize': self.text_size_sub})
        # plt.xticks(fontproperties='Times New Roman', size=self.text_size_sub)
        # plt.yticks(fontproperties='Times New Roman', size=self.text_size_sub)
        plt.pause(0.05)
        # plt.tight_layout()
        plt.savefig(self.path_out + str(self.file_name) + '_env.jpeg')

