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
    def __init__(self, element, loc, var, parameter, r, test_data, true_val,  expect_flag):
        self.data = element
        self.test_data = test_data
        self.loc = loc
        self.true_val = true_val
        self.distances = None
        self.samples = None
        self.squ_dis = None
        self.inter_neigh = None
        self.inter_squ_dis = None
        self.inter_dis = None
        self.G_value = None
        self.Q = None
        self.neighbors = [list() for i in range(len(self.test_data))]
        self.Var_S = var[0]
        self.Var = var[0]
        self.c_S = parameter[0][0]
        self.a_s = parameter[0][1]
        self.b_s = parameter[0][2]
        self.intercept_s = parameter[0][3]
        self.r = r
        self.expect_flag = expect_flag
        self.miss_flag = None

    def _average_distance(self):
        coordinate = np.array(self.loc)
        nabs = neighbors.NearestNeighbors(n_neighbors=31, algorithm='kd_tree').fit(coordinate)
        distances, indices = nabs.kneighbors(coordinate)
        self.distances = distances[:, 1:].tolist()
        self.neighbors = indices[:, 1:self.r[0] + 1].tolist()
        # self.neighbors = indices[:, 1:].tolist()

    def _get_inter_neigh_dis(self, center):
        shape = len(self.test_data)
        self.inter_neigh = list()
        self.inter_dis = list()
        self.inter_squ_dis = list()
        loc = self.loc.tolist()
        loc.append(self.test_data[center].tolist())
        coords = np.array(loc)
        nabs = neighbors.NearestNeighbors(n_neighbors=31, algorithm='kd_tree').fit(coords)
        distances, indices = nabs.kneighbors(coords)
        self.inter_neigh = (indices[:, 1:self.r[0] + 1].tolist())
        self.inter_dis = (distances[:, 1:].tolist())
        squ_dis = pdist(coords, metric='euclidean')
        self.inter_squ_dis = (squareform(squ_dis))



    def _get_squ_dis(self):
        self.squ_dis = pdist(self.loc, metric='euclidean')
        self.squ_dis = squareform(self.squ_dis)
        shape = len(self.data)
        self.miss_flag = np.array([0 for i in range(len(self.data))])
        for i in range(shape):
            if not np.isnan(self.data[i]):
                self.miss_flag[i] = 1
        # for i in range(len(self.distances)):
        #     for j in range(10 * self.r[0]):
        #         self.neighbors[i].append(self.indices[i][j])

    def _calculate_G_value(self, ls):
        Rls = self.intercept_s + self.a_s * exp(-self.b_s / (ls ** self.c_S  + 1e-9))
        Gst = self.Var - Rls
        return Gst

    # def _calculate_G_vector(self):
    #     shape = self.data.shape
    #     length = shape[0] * shape[1]
    #     G_value = np.zeros((length, length))
    #     for i in range(0, length):
    #         central_s = i // self.t_d
    #         central_t = i % self.t_d
    #         for j in range(i, length):
    #             neigh_s = j // self.t_d
    #             neigh_t = j % self.t_d
    #             G_value[i, j] = self._calculate_G_value(self.squ_dis[central_s][neigh_s],
    #                                                     abs(central_t - neigh_t))
    #             G_value[j, i] = G_value[i, j]
    #     self.G_value = G_value

    def _construct_G(self):
        central_G = np.zeros((len(self.samples), len(self.samples)))
        add_row = list()
        add_cow = list()
        new_row = list()
        new_cow = list()

        for i in range(0, len(self.samples)):
            for j in range(i, len(self.samples)):
                G_v = self._calculate_G_value(self.squ_dis[self.samples[i]][self.samples[j]])
                central_G[i, j] = G_v
                central_G[j, i] = G_v
            if self.expect_flag == 1:
                new_row.append(1)
                new_cow.append(1)
            else:
                new_row.append(1)
                new_cow.append(1)

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

    def _calculate_C_vector(self, central):
        C_value = list()
        self._get_inter_neigh_dis(central)
        # central_loc = central * self.t_d + central_t

        for j in range(len(self.inter_neigh[-1])):
            # if len(self.samples) >= self.r[0]:
            #     break
            if self.miss_flag[self.inter_neigh[-1][j]] == 1:
                # neigh_loc = self.neighbors[central][j] * self.t_d + l
                self.samples.append(self.inter_neigh[-1][j])
                # new_G = self.G_value[central_loc, neigh_loc]
                new_G = self._calculate_G_value(self.inter_squ_dis[-1][self.inter_neigh[-1][j]])
                C_value.append(new_G)

        C_value.append(1)
        return C_value

    def _get_samples_vector(self):
        Q = [0 for i in range(len(self.samples))]
        for j in range(len(self.samples)):
            Q[j] = (self.data[self.samples[j]])
        self.Q = Q

    def interpretation(self):
        interpretation_value = np.array(self.test_data, copy=True)
        result = list()
        # 注意这里
        shape = np.shape(self.test_data)[0]

        error_value = list()
        self._average_distance()
        self._get_squ_dis()
        # self._calculate_G_vector()
        # time_error = np.zeros((shape, 1))
        # t_square_error = np.zeros((shape[1], 1))
        # spatial_errors = np.zeros((shape, 1))
        # time_errors = [list() for i in range(self.t_d)]
        # s_square_errors = np.zeros((shape, 1))
        # t_square_errors = [list() for i in range(self.t_d)]
        spatial_error = list()
        for i in range(shape):

            square_error = list()
            self.samples = list()
            central_C = self._calculate_C_vector(i)
            central_G = self._construct_G()
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
            interpretation_value[i] = mid_inter_value[0][0]
            result.append(interpretation_value[i][0])
            # print("iter{0}, true_value:{1}, pred_value:{2}".format(i, self.true_val[i],interpretation_value[i][0]))
            error = abs(self.true_val[i] - interpretation_value[i][0])
            error_value.append(error)
            spatial_error.append(error)
            #     # time_errors[j].append(error)
            #     # t_square_errors[j].append(error**2)
            #     square_error.append(error ** 2)
            #     print(time.process_time())
            # if len(spatial_error) != 0:
            #     spatial_errors[i] = np.mean(spatial_error)
            #     s_square_errors[i] = np.sqrt(np.mean(square_error))
            # else:
            #     spatial_errors[i] = 0
        # for j in range(self.t_d):
        #     time_error[j] = np.mean(time_errors[j])
        #     t_square_error[j] = np.sqrt(np.mean(t_square_errors[j]))
        RMSE = 0
        sum = 0
        for err in spatial_error:
            sum += np.power(err, 2)
        RMSE = np.sqrt(sum / shape)

        return np.array(result), RMSE

