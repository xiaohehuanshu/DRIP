import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as mpl
import openpyxl
import os


class GraphPlan:
    def __init__(self, layout_info, file_name="", path_out="", if_save=True):
        self.layout_info = layout_info
        self.path_out = path_out
        self.file_name = file_name
        self.text_size = 18
        self.text_size_sub = 18
        self.if_save = if_save

        if "dinning" in self.layout_info.columns:
            self.layout_info = self.layout_info.rename(columns={"dinning": "dining"})
        if "dinner" in layout_info.columns:
            self.layout_info = self.layout_info.rename(columns={"dinner": "dining"})
        if "toilet1" in self.layout_info.columns:
            self.layout_info = self.layout_info.rename(columns={"toilet1": "bath1"})
        if "toilet2" in self.layout_info.columns:
            self.layout_info = self.layout_info.rename(columns={"toilet2": "bath2"})

        self.sequence = [
            "boundary",
            "black1",
            "black2",
            "black3",
            "black4",
            "entrance",
            "hallway",
            "white_south",
            "white_north",
            "white_east",
            "white_west",
            "white_m1",
            "white_m2",
            "white_m3",
            "white_m4",
            "room1",
            "room2",
            "room3",
            "room4",
            "living",
            "dining",
            "bath1",
            "bath2",
            "bath3",
            "bath1_sub",
            "bath2_sub",
            "kitchen",
            "storage",
            "storeroom",
            "gate",
            "courtyard",
            "staircase",
            "porch",
            "study_room",
            "blank",
            "entrance_sub",
            "garage"
        ]

        self.graph_color = {
            "boundary": "#FFE27D",
            "entrance": "whitesmoke",
            "hallway": "#FFE27D",
            "white_south": "whitesmoke",
            "white_north": "whitesmoke",
            "white_east": "whitesmoke",
            "white_west": "whitesmoke",
            "white_m1": "whitesmoke",
            "white_m2": "whitesmoke",
            "white_m3": "whitesmoke",
            "white_m4": "whitesmoke",
            "black1": "darkgrey",
            "black2": "darkgrey",
            "black3": "darkgrey",
            "black4": "darkgrey",
            "room1": "#F3A169",
            "room2": "#F3A169",
            "room3": "#F3A169",
            "room4": "lightgray",
            "living": "#FFE27D",
            "dining": "#C5D5AE",
            "bath1": "#6483A6",
            "bath2": "#90AEC4",
            "bath3": "#90AEC4",
            "bath1_sub": "#90AEC4",
            "bath2_sub": "#90AEC4",
            "kitchen": "#80A6AF",
            "storage": "#90AEC4",
            "storeroom": "#90AEC4",
            "gate": "whitesmoke",
            "courtyard": "whitesmoke",
            "staircase": "#8474A0",
            "porch": "whitesmoke",
            "study_room": "#F3A169",
            "blank": "whitesmoke",
            "entrance_sub": "whitesmoke",
            "garage": "gray",
        }

        self.text_trans = {
            "entrance": "Entrance",
            "entrance_sub": "Entrance_sub",
            "hallway": "hallway",
            "white_south": "Light",
            "white_north": "Light",
            "white_east": "Light",
            "white_west": "Light",
            "white_m1": "Light1",
            "white_m2": "Light2",
            "white_m3": "Light3",
            "white_m4": "Light4",
            "black1": "",
            "black2": "",
            "black3": "",
            "black4": "",
            "room1": "Room1",
            "room2": "Room2",
            "room3": "Room3",
            "room4": "Room4",
            "study_room": "Study",
            "living": "Living",
            "dining": "Dining",
            "bath1": "Bath1",
            "bath2": "Bath2",
            "bath3": "Bath3",
            "bath1_sub": "Bath1_sub",
            "bath2_sub": "Bath2_sub",
            "kitchen": "Kitchen",
            "staircase": "Stair",
            "storeroom": "Storeroom",
            "storage": "Storage",
            "pub": "",
            "gate": "Gate",
            "courtyard": "Courtyard",
            "porch": "Porch",
            "blank": "Blank",
            "garage": "Garage",
        }

        self.names_exisit = [i for i in self.sequence if i in self.layout_info]
        self.layout_info = self.layout_info[self.names_exisit]


    def draw_plan(self, action_choose, ax):
        mpl.rcParams["font.sans-serif"] = ["Times New Roman"]
        mpl.rcParams["axes.unicode_minus"] = False

        name_list = self.layout_info.columns.values.tolist()
        lis_exclude = ["living", "dining"]

        if "entrance" in name_list:
            name_list.remove("entrance")
            name_list = name_list + ["entrance"]

        for i, item in enumerate(name_list):
            tmp = self.layout_info.loc["rec", item]
            # print(item, tmp)
            x, y = tmp.exterior.xy
            # print(x,y)
            # print('----------')
            ax.plot(x, y, color="black", lw=3)
            # print(item)
            ax.fill(x, y, color=self.graph_color[item], alpha=0.8)

            # if item not in ['white_m1', 'whtie_m2', 'white_m3']:
            if item != "boundary":
                center = tmp.centroid.coords[0]
                cx = center[0]
                cy = center[1]
                ax.text(
                    cx,
                    cy,
                    s="%s" % (self.text_trans[item]),
                    color="black",
                    verticalalignment="center",
                    horizontalalignment="center",
                    size=self.text_size,
                )

        if action_choose is not None:
            x_, y_, w_, d_ = action_choose.values
            x1, y1 = x_, y_
            x2, y2 = (x_ + w_), (y_ + d_)
            ax.scatter(x1, y1, color="red", label="First Point", zorder=5)
            ax.text(x1, y1, f"({x1:.0f}, {y1:.0f})", fontsize=12, zorder=5)
            ax.scatter(x2, y2, color="blue", label="Second Point", zorder=5)
            ax.text(x2, y2, f"({x2:.0f}, {y2:.0f})", fontsize=12, zorder=5)

        ax.set_xlim([-3000, 19000])
        ax.set_ylim([-3000, 19000])

        ax.grid(True)
        if self.if_save:
            plt.savefig(self.path_out + str(self.file_name) + ".jpg", dpi=300)

    def draw_plan_all(self):
        mpl.rcParams["font.sans-serif"] = ["Times New Roman"]
        mpl.rcParams["axes.unicode_minus"] = False

        plt.figure("layout", figsize=(9, 9))
        ax1 = plt.subplot(1, 1, 1)
        plt.cla()

        name_list = self.layout_info.columns.values.tolist()[1:]
        # name_list.reverse()
        name_list = ["boundary"] + name_list
        if "entrance" in name_list:
            name_list.remove("entrance")
            name_list = name_list + ["entrance"]

        for i, item in enumerate(name_list):
            tmp = self.layout_info[item]
            X, Y, W, H = tmp.values
            if W != 0 and H != 0:
                if item == "boundary":
                    rect = plt.Rectangle((X, Y), W, H, fill=True, facecolor=self.graph_color[item], alpha=1, lw=3)
                    ax1.add_patch(rect)
                    rect1 = plt.Rectangle((X, Y), W, H, fill=False, edgecolor="black", alpha=0.9, lw=3)
                    ax1.add_patch(rect1)
                    rect.set_label(item)
                else:
                    rect = plt.Rectangle((X, Y), W, H, fill=True, facecolor=self.graph_color[item], alpha=1, lw=3)
                    ax1.add_patch(rect)
                    rect1 = plt.Rectangle((X, Y), W, H, fill=False, edgecolor="black", alpha=0.9, lw=3)
                    ax1.add_patch(rect1)
                    rect.set_label(item)

                    plt.text(
                        X + W / 2,
                        Y + H / 2,
                        s="%s" % (self.text_trans[item]),
                        color="black",
                        verticalalignment="center",
                        horizontalalignment="center",
                        size=self.text_size,
                    )

        plt.xlim([-3000, 18000])
        plt.ylim([-3000, 18000])
        # plt.title(self.file_name, fontdict={'fontsize': self.text_size_sub})
        plt.xticks(fontproperties="Times New Roman", size=self.text_size_sub)
        plt.yticks(fontproperties="Times New Roman", size=self.text_size_sub)
        plt.pause(0.05)
        plt.tight_layout()
        plt.savefig(self.path_out + str(self.file_name) + ".jpg")

