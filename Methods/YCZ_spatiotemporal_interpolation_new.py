# -*-coding=utf-8-*-
#
"""
@Author:Jieyang
@Time:2021-5-24
@Description:Yangchizhong interpolation for spatiotemporal data
"""

import numpy as np
from math import exp
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import time
from sklearn import neighbors


class YCZ_interpretation:
    def __init__(self, element, loc, var, parameter, r, test_data, mean_ratio, expect_flag):
        self.data = element
        self.test_data = test_data
        self.loc = loc
        self.distances = None
        self.samples = None
        self.squ_dis = None
        self.G_value = None
        self.Q = None
        self.neighbors = [list() for i in range(len(self.test_data))]
        self.Var_S = var[0]
        self.Var_T = var[1]
        self.Var = var[2]
        self.c_S = parameter[0][0]
        self.c_T = parameter[1][0]
        self.a_s = parameter[0][1]
        self.b_s = parameter[0][2]
        self.intercept_s = parameter[0][3]
        self.a_t = parameter[1][1]
        self.b_t = parameter[1][2]
        self.intercept_t = parameter[1][3]
        self.k1 = (self.Var - self.Var_T) / self.Var_S
        self.k2 = (self.Var - self.Var_S) / self.Var_T
        self.k3 = (self.Var_S + self.Var_T - self.Var) / (self.Var_T * self.Var_S)
        self.r = r
        self.t_d = len(self.data[0])
        self.mean_ratio = mean_ratio
        self.expect_flag = expect_flag
        self.miss_flag = None

    def _average_distance(self):
        coordinate = np.array(self.loc)
        nabs = neighbors.NearestNeighbors(n_neighbors=31, algorithm='kd_tree').fit(coordinate)
        distances, indices = nabs.kneighbors(coordinate)
        self.distances = distances[:, 1:].tolist()
        self.neighbors = indices[:, 1:self.r[0] + 1].tolist()
        # self.neighbors = indices[:, 1:].tolist()

    def _get_squ_dis(self):
        self.squ_dis = pdist(self.loc, metric='euclidean')
        self.squ_dis = squareform(self.squ_dis)
        shape = self.data.shape
        self.miss_flag = np.zeros(shape, dtype=np.int)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if not np.isnan(self.data[i, j]):
                    self.miss_flag[i, j] = 1
        # for i in range(len(self.distances)):
        #     for j in range(10 * self.r[0]):
        #         self.neighbors[i].append(self.indices[i][j])

    def _calculate_G_value(self, ls, lt):
        if ls == 0 and lt == 0:
            Gst = self.Var
        elif ls == 0:
            Rlt = self.intercept_t + self.a_t * exp(-self.b_t / (lt ** self.c_T))
            Gst = self.Var - Rlt
        elif lt == 0:
            Rls = self.intercept_s + self.a_s * exp(-self.b_s / (ls ** self.c_S))
            Gst = self.Var - Rls
        else:
            Rls = self.intercept_s + self.a_s * exp(-self.b_s / (ls ** self.c_S))
            Rlt = self.intercept_t + self.a_t * exp(-self.b_t / (lt ** self.c_T))
            Gls = self.Var_S - Rls
            Glt = self.Var_T - Rlt
            Gst = self.k1 * Gls + self.k2 * Glt + self.k3 * Gls * Glt
        return Gst

    def _calculate_G_vector(self):
        shape = self.data.shape
        length = shape[0] * shape[1]
        G_value = np.zeros((length, length))
        for i in range(0, length):
            central_s = i // self.t_d
            central_t = i % self.t_d
            for j in range(i, length):
                neigh_s = j // self.t_d
                neigh_t = j % self.t_d
                G_value[i, j] = self._calculate_G_value(self.squ_dis[central_s][neigh_s],
                                                        abs(central_t - neigh_t))
                G_value[j, i] = G_value[i, j]
        self.G_value = G_value

    def _construct_G(self, central, central_t):
        central_G = np.zeros((len(self.samples), len(self.samples)))
        add_row = list()
        add_cow = list()
        new_row = list()
        new_cow = list()

        for i in range(0, len(self.samples)):
            for j in range(i, len(self.samples)):
                G_v = self._calculate_G_value(self.squ_dis[self.samples[i][0]][self.samples[j][0]],
                                                        abs(self.samples[i][1] - self.samples[j][1]))
                central_G[i, j] = G_v
                central_G[j, i] = G_v
            if self.expect_flag == 1:
                new_row.append(self.mean_ratio[central, self.samples[i][0]])
                new_cow.append(self.mean_ratio[central, self.samples[i][0]])
            else:
                new_row.append(self.mean_ratio[central_t, self.samples[i][1]])
                new_cow.append(self.mean_ratio[central_t, self.samples[i][1]])

        new_cow.append(0)
        add_row.append(new_row)
        add_cow.append(new_cow)
        add_row = np.array(add_row)
        add_cow = np.array(add_cow)

        # add_row = [[1 for i in range(len(central_G))] for j in range(1)]
        # add_row = np.array(add_row)
        # add_cow = [[1 for i in range(len(central_G) + 1)] for j in range(1)]
        # add_cow = np.array(add_cow)

        central_G = np.r_[central_G, add_row]
        central_G = np.c_[central_G, add_cow.T]
        central_G[len(central_G) - 1][len(central_G) - 1] = 0

        return central_G

    def _calculate_C_vector(self, central, central_t):
        C_value = list()
        # central_loc = central * self.t_d + central_t
        shape = self.data.shape
        for l in range(shape[1]):
            if 0 < abs(central_t - l) <= self.r[1]:
                if self.miss_flag[central, l] == 1:
                    # neigh_loc = central * self.t_d + l
                    self.samples.append((central, l))
                    new_G = self._calculate_G_value(self.squ_dis[central][central],
                                                        abs(central_t - l))
                    C_value.append(new_G)
            # if len(self.samples) >= self.r[0]:
            #     break

        for j in range(len(self.neighbors[central])):
            # if len(self.samples) >= self.r[0]:
            #     break
            for l in range(shape[1]):
                if 0 <= abs(central_t - l) <= self.r[1]:
                    if self.miss_flag[self.neighbors[central][j], l] == 1:
                        # neigh_loc = self.neighbors[central][j] * self.t_d + l
                        self.samples.append((self.neighbors[central][j], l))
                        # new_G = self.G_value[central_loc, neigh_loc]
                        new_G = self._calculate_G_value(self.squ_dis[central][self.neighbors[central][j]],
                                                        abs(central_t - l))
                        C_value.append(new_G)
        C_value.append(1)
        return C_value

    def _get_samples_vector(self):
        Q = [0 for i in range(len(self.samples))]
        for j in range(len(self.samples)):
            Q[j] = (self.data[self.samples[j][0]][self.samples[j][1]])
        self.Q = Q

    def interpretation(self):
        interpretation_value = np.array(self.data, copy=True)
        shape = self.test_data.shape
        error_value = list()
        self._average_distance()
        self._get_squ_dis()
        # self._calculate_G_vector()
        time_error = np.zeros((shape[1], 1))
        t_square_error = np.zeros((shape[1], 1))
        spatial_errors = np.zeros((shape[0], 1))
        time_errors = [list() for i in range(self.t_d)]
        s_square_errors = np.zeros((shape[0], 1))
        t_square_errors = [list() for i in range(self.t_d)]

        for i in range(0, shape[0]):
            spatial_error = list()
            square_error = list()
            for j in range(shape[1]):
                if np.isnan(self.data[i, j]):
                    self.samples = list()
                    central_C = self._calculate_C_vector(i, j)
                    central_G = self._construct_G(i, j)
                    self._get_samples_vector()
                    central_C = np.array(central_C)
                    central_C = np.reshape(central_C, (len(central_C), 1))
                    # add_row = [[1 for i in range(len(central_G))] for j in range(1)]
                    # add_row = np.array(add_row)
                    # add_cow = [[1 for i in range(len(central_G) + 1)] for j in range(1)]
                    # add_cow = np.array(add_cow)
                    #
                    # central_G = np.r_[central_G, add_row]
                    # central_G = np.c_[central_G, add_cow.T]
                    # central_G[len(central_G) - 1][len(central_G) - 1] = 0
                    if np.linalg.det(central_G) != 0:
                        central_G = np.linalg.inv(central_G)
                    else:
                        central_G = np.linalg.pinv(central_G)

                    self.Q = np.array(self.Q)
                    self.Q = np.reshape(self.Q, (len(self.Q), 1))

                    mid = np.dot(central_G, central_C)
                    mid = np.transpose(mid[:-1])
                    a = np.sum(mid)
                    mid_inter_value = np.dot(mid, self.Q)
                    interpretation_value[i, j] = mid_inter_value[0][0]
                    error = abs(self.test_data[i, j] - interpretation_value[i, j])
                    error_value.append(error)
                    spatial_error.append(error)
                    time_errors[j].append(error)
                    t_square_errors[j].append(error**2)
                    square_error.append(error ** 2)
                    print(time.process_time())
            if len(spatial_error) != 0:
                spatial_errors[i] = np.mean(spatial_error)
                s_square_errors[i] = np.sqrt(np.mean(square_error))
            else:
                spatial_errors[i] = 0
        for j in range(self.t_d):
            time_error[j] = np.mean(time_errors[j])
            t_square_error[j] = np.sqrt(np.mean(t_square_errors[j]))

        return interpretation_value, error_value, spatial_errors, time_error, s_square_errors, t_square_error

