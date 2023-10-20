# -*- coding = utf-8 -*-
"""
@author: -
@Time: 2022-3-14
@paper: YangChizhong filtering
"""
#

import numpy as np
from sklearn import neighbors
from math import exp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import simps
import copy


# Generalized Yang Chizhong
class YCZ_filter:
    def __init__(self, element, loc):
        self.element = element
        self.loc = loc
        self.distances = None
        self.indices = None
        self.ave_dis = 0
        self.c = 0
        self.filtered_value = copy.deepcopy(element)
        self.observe_error = 0
        self.neighbors = [list() for i in range(len(self.element))]
        self.weights = [list() for i in range(len(self.element))]

    def _average_distance(self):
        coordinate = np.array(self.loc)
        nabs = neighbors.NearestNeighbors(n_neighbors=20, algorithm='kd_tree').fit(coordinate)
        distances, indices = nabs.kneighbors(coordinate)
        self.ave_dis = np.mean(distances[:, 1:5])
        self.distances = distances[:, 1:].tolist()
        self.indices = indices[:, 1:].tolist()

    def _search_neighbor(self):
        for i in range(len(self.distances)):
            for j in range(len(self.distances[i])):
                if self.distances[i][j] < (self.ave_dis * (2 ** 0.5)):
                    self.neighbors[i].append(self.indices[i][j])

    def _calculate_weight(self):
        for i in range(len(self.neighbors)):
            for nei in self.neighbors[i]:
                index_1 = self.indices[i].index(nei)
                weight = 4 * exp(-0.693 * ((self.distances[i][index_1] / self.ave_dis) ** 2))
                self.weights[i].append(weight)

    def _calculate_R_value(self):
        filter_error = np.var(self.filtered_value)
        return self.observe_error - filter_error

    def plot_random_fuc(self):
        Y = [0]
        X = [0]
        self._average_distance()
        self._search_neighbor()
        self._calculate_weight()

        self.observe_error = np.var(self.element)

        for i in range(20):
            new_filter_value = list()
            X.append((i + 1) * self.ave_dis * (2 ** 0.5))
            for j in range(len(self.neighbors)):
                if sum(self.weights[j]) < 12:
                    new_element = (16 - sum(self.weights[j])) * self.filtered_value[j]
                    for k in range(len(self.neighbors[j])):
                        nei = self.neighbors[j][k]
                        new_element += self.filtered_value[nei] * self.weights[j][k]
                    new_filter_value.append(new_element / 16)
                else:
                    new_element = 4 * self.filtered_value[j]
                    for k in range(len(self.neighbors[j])):
                        nei = self.neighbors[j][k]
                        new_element += self.filtered_value[nei] * self.weights[j][k]
                    new_filter_value.append(new_element / (sum(self.weights[j]) + 4))
            self.filtered_value = new_filter_value
            Y.append(self._calculate_R_value())
        y = np.array(Y)
        plt.plot(X, Y)
        plt.show(block=True)
        cc = simps(y, dx=int(self.ave_dis * (2 ** 0.5)))
        new_c = cc / (X[-1] * np.mean(self.observe_error))
        # self.intercept = (final_X[1] * final_Y[2] - final_X[2] * final_Y[1]) / (final_X[1] - final_X[2])
        # new_c = 2 * (1 - new_c)
        if 0.5 <= new_c <= 1.5:
            self.c = new_c
        elif new_c < 0.5:
            self.c = 0.5
        else:
            self.c = 1.5
        repeat_time = 0
        for i in range(1, (len(Y) - 1)):
            if ((Y[i + 1] - Y[i]) / Y[i + 1]) <= 0.08:
                repeat_time = i + 1
                break
        return repeat_time, X, Y

    def solve_filtered_value(self, repeated_time):
        self._average_distance()
        self._search_neighbor()
        self._calculate_weight()
        self.filtered_value = copy.deepcopy(self.element)
        #
        for i in range(repeated_time):
            new_filter_value = list()
            for j in range(len(self.neighbors)):
                if sum(self.weights[j]) < 12:
                    new_element = (16 - sum(self.weights[j])) * self.filtered_value[j]
                    for k in range(len(self.neighbors[j])):
                        nei = self.neighbors[j][k]
                        new_element += self.filtered_value[nei] * self.weights[j][k]
                    new_filter_value.append(new_element / 16)
                else:
                    new_element = 4 * self.filtered_value[j]
                    for k in range(len(self.neighbors[j])):
                        nei = self.neighbors[j][k]
                        new_element += self.filtered_value[nei] * self.weights[j][k]
                    new_filter_value.append(new_element / (4 + sum(self.weights[j])))
            self.filtered_value = new_filter_value

        return np.array(self.filtered_value)

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

        # plt.scatter(X, Y)
        # plt.plot(X, y_fit, color='red', linewidth=1.0)
        # plt.show()
        return a, b, inter

    def fit_spatial_function(self):

        repeat_time, final_X, final_Y = self.plot_random_fuc()
        a, b, inter = self.parameter_fitting(final_X, final_Y)
        return [self.c, a, b, inter], np.mean(self.observe_error), final_X[-1]



#
