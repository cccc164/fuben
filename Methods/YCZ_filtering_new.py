# -*- coding = utf-8 -*-
#
"""
@Auther:Jieyang
@Time: 2021-8-30
@Description:Yangchizhong filtering for spatial data
"""

import numpy as np
from sklearn import neighbors
from math import exp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import simps
import copy


class neighbor(object):
    def __init__(self, num):
        self.weight = [list() for i in range(num)]
        self.neighbor = [list() for i in range(num)]


class YCZ_filter:
    def __init__(self, element, loc):
        self.data = element
        self.loc = loc
        self.filtered_value = copy.deepcopy(element)
        self.ave_dis = 0
        self.observe_error = list()
        self.neighbors = list()
        self.c = 1.5
        self.intercept = 0

    def _calculate_neighbor(self):
        all_ave_dis = list()
        print(self.data)
        for i in range(len(self.data)):
            coordinate = self.loc[i]
            coordinate = np.array(coordinate).reshape(1, -1)
            nabs = neighbors.NearestNeighbors(n_neighbors=len(coordinate), algorithm='kd_tree').fit(np.array(coordinate))
            distances, indices = nabs.kneighbors(coordinate)
            ave_dis = np.mean(distances[:, 1:5])
            all_ave_dis.append(ave_dis)
            distances = distances[:, 1:].tolist()
            indices = indices[:, 1:].tolist()
            new_neighbor = neighbor(len(self.loc[i]))
            for p in range(len(distances)):
                for q in range(len(distances[p])):
                    if distances[p][q] < (ave_dis * (2 ** 0.5)):
                        new_neighbor.neighbor[p].append(indices[p][q])
                    else:
                        break
                # if len(new_neighbor.neighbor[p]) == 0:
                #     null_num += 1
                #     for j in range(2):
                #         new_neighbor.neighbor[p].append(indices[p][j])
            # print(null_num)
            for p in range(len(new_neighbor.neighbor)):
                for nei in new_neighbor.neighbor[p]:
                    index_1 = indices[p].index(nei)
                    weight = 4 * exp(-0.693 * ((distances[p][index_1] / ave_dis) ** 2))
                    new_neighbor.weight[p].append(weight)
            self.neighbors.append(new_neighbor)
        self.ave_dis = np.mean(all_ave_dis)

    def _calculate_R_value(self):
        filter_error = list()
        for i in range(len(self.data)):
            error = self.observe_error[i] - np.var(self.filtered_value[i])
            filter_error.append(error)
        return np.mean(filter_error)

    def _get_filter_data(self):
        self._calculate_neighbor()
        Y = [[0] for i in range(len(self.neighbors))]
        X = [[0] for i in range(len(self.neighbors))]

        for t in range(len(self.data)):
            self.observe_error.append(np.var(self.data[t]))

        for i in range(20):
            for t in range(len(self.data)):
                X[t].append(i + 1)
                new_filter_value = list()
                for j in range(len(self.neighbors[t].neighbor)):
                    if sum(self.neighbors[t].weight[j]) < 12:
                        new_element = (16 - sum(self.neighbors[t].weight[j])) * self.filtered_value[t][j]
                        for k in range(len(self.neighbors[t].neighbor[j])):
                            nei = self.neighbors[t].neighbor[j][k]
                            new_element += self.filtered_value[t][nei] * self.neighbors[t].weight[j][k]
                        new_filter_value.append(new_element / 16)
                    else:
                        new_element = 4 * self.filtered_value[t][j]
                        for k in range(len(self.neighbors[t].neighbor[j])):
                            nei = self.neighbors[t].neighbor[j][k]
                            new_element += self.filtered_value[t][nei] * self.neighbors[t].weight[j][k]
                        final_ele = new_element / (4 + sum(self.neighbors[t].weight[j]))
                        new_filter_value.append(final_ele)
                self.filtered_value[t] = new_filter_value
                error = self.observe_error[t] - np.var(self.filtered_value[t])
                Y[t].append(error)

        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # # ax.scatter(X[17], Y[17], c='b', marker='*', s=100, label='k_value = 18')
        # # plt.legend(bbox_to_anchor=(0.95, 0.9), ncol=1, borderaxespad=0, fontsize='large')
        # ax.plot(X, Y, c='r')
        # plt.xlabel('L', size=15)
        # plt.ylabel('R(L)', size=15)
        # plt.tick_params(labelsize=10)
        # plt.show()
        repeat_time = 0
        for i in range(1, 19):
            flag = 1
            for t in range(len(self.data)):
                if ((Y[t][i + 1] - Y[t][i]) / Y[t][i + 1]) > 0.08:
                    flag = 0
                    break
                if t == (len(self.data) - 1) and flag == 1:
                    repeat_time = i + 1
                    break
        final_X = list()
        final_Y = list()
        for i in range(repeat_time + 1):
            final_X.append(i * self.ave_dis * (2 ** 0.5))
            sum_Y = 0
            for t in range(len(Y)):
                sum_Y += Y[t][i]
            final_Y.append(sum_Y / len(Y))
        y = np.array(final_Y)
        cc = simps(y, dx=int(self.ave_dis * (2 ** 0.5)))
        new_c = cc / (final_X[-1] * np.mean(self.observe_error))
        self.intercept = (final_X[1] * final_Y[2] - final_X[2] * final_Y[1]) / (final_X[1] - final_X[2])
        # new_c = 2 * (1 - new_c)
        if 0.5 <= new_c <= 1.5:
            self.c = new_c
        elif new_c < 0.5:
            self.c = 0.5
        else:
            self.c = 1.5
        # self.c = new_c

        return final_X, final_Y

    def shaffer(self, x, a, b, d):
        y = np.power(x, self.c)
        y = -b / (y + 1e-8)
        result = d + np.exp(y) * a
        # result = np.exp(y) * a

        return result

    def parameter_fitting(self, X, Y):
        pop, cov = curve_fit(self.shaffer, X, Y, [1000, 1, 0])
        a = pop[0]
        b = pop[1]
        inter = pop[2]
        y_fit = self.shaffer(X, a, b, inter)

        plt.scatter(X, Y)
        plt.plot(X, y_fit, color='red', linewidth=1.0)
        plt.show()
        return a, b, inter

    def fit_spatial_function(self):
        final_X, final_Y = self._get_filter_data()
        a, b, inter = self.parameter_fitting(final_X, final_Y)
        return [self.c, a, b, inter], np.mean(self.observe_error), final_X[-1]
#
