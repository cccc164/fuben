# -*- coding = utf-8 -*-
#
# 预处理数据（包括 等距对数比变换及归一化数据）
import pandas as pd
import math
import numpy as np


class Data_deal:
    def __init__(self, element_list):
        self.element_list = element_list
        self.cor = list()

    def _isometric_log_transformation(self, row):
        ele_multi = 1
        result_list = [0.0 for x in range(0, len(self.element_list))]
        for i in range(0, len(self.element_list) - 1):
            ele_multi *= row[self.element_list[i]]
            result = math.log(math.pow(ele_multi, 1 / (i + 1)) / row[self.element_list[i + 1]])
            result *= math.pow((i + 1) / (i + 2), 0.5)
            result_list[i] = result

        # for i in range(0, len(self.element_list)):
        #     ele_multi *= row[self.element_list[i]]
        # ele_multi = math.pow(ele_multi, 1 / (len(self.element_list)))
        # # 中心对数比变换
        # for i in range(len(self.element_list)):
        #     result_list[i] = math.log(row[self.element_list[i]] / ele_multi)
        # 反变换
        # inverse_matrix = np.zeros((len(self.element_list), len(self.element_list)-1))
        # for i in range(len(self.element_list) - 1):
        #     for j in range(i + 1):
        #         inverse_matrix[j, i] = math.pow((i + 1) / (i + 2), 0.5) / (i + 1)
        #     inverse_matrix[i + 1, i] = -math.pow((i + 1) / (i + 2), 0.5)
        # ilr_result = np.dot(np.array(result_list)[0:38], inverse_matrix.T)
        # clr_c = self.ilr2clr(np.array(result_list))
        # clr_result = np.dot(inverse_matrix, ilr_result)
        # clr_c = 0
        # 对数比变换
        # ilr_result = [0.0 for x in range(0, len(self.element_list) - 1)]
        # for i in range(0, len(self.element_list) - 1):
        #     ele_multi *= result_list[i]
        #     result = math.log(math.pow(ele_multi, 1 / (i + 1)) / result_list[i + 1])
        #     result *= math.pow((i + 1) / (i + 2), 0.5)
        #     ilr_result[i] = result

        # for i in range(0, len(self.element_list) - 1):
        #     ele_multi /= row[self.element_list[i]]
        #     result = math.log(row[self.element_list[i]] / math.pow(ele_multi, 1 / (len(self.element_list) - 1 - i)))
        #     result *= math.pow((len(self.element_list) - 1 - i) / (len(self.element_list) - i), 0.5)
        #     result_list[i] = result
        # 反变换
        # inverse_matrix = np.zeros((len(self.element_list), len(self.element_list)))
        # for i in range(len(self.element_list) - 1):
        #     for j in range(i + 1):
        #         inverse_matrix[j, i] = math.pow((i + 1) / (i + 2), 0.5) / (i + 1)
        #     inverse_matrix[i + 1, i] = -math.pow((i + 1) / (i + 2), 0.5)
        # new_result = np.dot(inverse_matrix, np.array(result_list))

        # for i in range(0, len(self.element_list) - 1):
        #     ele_multi /= row[self.element_list[i]]
        #     result = math.log(row[self.element_list[i]] / math.pow(ele_multi, 1 / (len(self.element_list) - 1 - i)))
        #     result *= math.pow((len(self.element_list) - 1 - i) / (len(self.element_list) - i), 0.5)
        #     result_list[i] = result
        # inverse_matrix = np.zeros((len(self.element_list), len(self.element_list)))
        # for i in range(len(self.element_list) - 1):
        #     for j in range(i + 1):
        #         inverse_matrix[j, i] = math.pow((i + 1) / (i + 2), 0.5) / (i + 1)
        #     inverse_matrix[i + 1, i] = -math.pow((i + 1) / (i + 2), 0.5)
        # new_result = np.dot(np.array(result_list), inverse_matrix)
        # result_list = clr_result.tolist()
        return result_list




    def _normalize_element_value(self, result_df, chosen_list):
        ele_num = len(chosen_list)

        for i in range(ele_num):
            max_value = max(result_df[chosen_list[i]])
            min_value = min(result_df[chosen_list[i]])
            for j in range(len(result_df[chosen_list[i]])):
                result_df[chosen_list[i]][j] = (result_df[chosen_list[i]][j] - min_value) \
                                                     / (max_value - min_value)

    def pre_process_data(self, file_name, chosen_list):
        df = pd.read_csv(file_name)
        result_dic = {}
        cor = list()
        element = list()
        for i in range(0, len(self.element_list) - 1):
            result_dic[self.element_list[i]] = []
        result_dic[chosen_list[-1]] = list()
        result_df = pd.DataFrame(result_dic)
        for index, row in df.iterrows():
            trans_list = self._isometric_log_transformation(row)
            append_dic = {chosen_list[-1]: row[chosen_list[-1]]}
            for i in range(0, len(trans_list)):
                ele_name = self.element_list[i]
                trans_ele = trans_list[i]
                append_dic[ele_name] = trans_ele
            cor.append([row['Coor_X'], row['Coor_Y']])
            result_df = result_df.append(append_dic, ignore_index=True)
            # result_df = pd.concat([result_df, append_dic], axis=1)
        self._normalize_element_value(result_df, chosen_list)

        # clr_result
        clr_result = np.zeros((result_df.shape))
        for i in range(len(result_df)):
            clr_result[i] = ilr2clr(result_df.loc[i].values[0:result_df.shape[1] - 1])
        for ele in chosen_list:
            element.append(result_df[ele])
        return element, cor, clr_result

def ilr2clr(ilr_result):
        dim = ilr_result.shape[0] + 1
        # 反变换
        inverse_matrix = np.zeros((dim, dim-1))
        for i in range(dim - 1):
            for j in range(i + 1):
                inverse_matrix[j, i] = math.pow((i + 1) / (i + 2), 0.5) / (i + 1)
            inverse_matrix[i + 1, i] = -math.pow((i + 1) / (i + 2), 0.5)
        clr_result = np.dot(np.array(ilr_result), inverse_matrix.T)
        return clr_result
if __name__ == "__main__":
    Data_deal.pre_process_data('Sample_NoNan_Pj_RS.csv', ['Ag', 'Au'])


