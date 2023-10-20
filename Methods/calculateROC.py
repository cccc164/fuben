import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import os

def ROC_new():
    non_mine = np.zeros((102, 82)) + 1
    mine_dh = pd.read_csv('mc/mine_dh.csv')
    mine_27 = np.zeros((102, 82))
    mine_all = pd.read_csv('mc/mine_1.csv')
    mine_1 = np.zeros((102, 82))
    score_39 = np.array(Image.open('mc/kernel_size/score3_128_7_500.tif'))
    # score_39_z = ((score_39 - np.mean(score_39)) / np.var(score_39))
    # arr2raster(score_39_z, 'mc/mineral occurrence.shp', 'mc/result_new/score3_128_5_z.tif')
    score_21 = np.array(Image.open('mc/kernel_size/score3_128_7_300_phase_1000_7.tif'))
    mine_dh_x = mine_dh['dx'].values
    mine_dh_y = mine_dh['dy'].values
    mine_x = mine_all['dx'].values
    mine_y = mine_all['dy'].values
    x_min = 40476796.08
    y_max = 4170032.922

    for x in range(82):
        for y in range(102):
            x_left = x_min + x * 1000
            y_up = y_max - y * 1000
            for i in range(len(mine_x)):
                if np.sqrt(np.power((mine_x[i] - x_left), 2) + np.power((mine_y[i] - y_up), 2)) < 15000:
                    non_mine[y][x] = 0
            for i in range(len(mine_dh)):
                if (mine_dh_x[i] >= x_left and mine_dh_y[i] <= y_up
                        and mine_dh_x[i] < (x_left + 1000) and mine_dh_y[i] > (y_up - 1000)):
                    mine_27[y][x] = 1
                    mine_1[y][x] = 1

    mine_random = np.argwhere(mine_27 == 1)
    mine_1_index = np.argwhere(mine_1 == 1)
    non_mine_index = np.argwhere(non_mine == 1)

    np.random.shuffle(mine_1_index)
    np.random.shuffle(non_mine_index)

    non_mine_random = non_mine_index[0:27]
    mine_1_random = mine_1_index[0:27]

    true = np.zeros(27 * 2)
    pred_21 = np.zeros(27 * 2)
    pred_39 = np.zeros(27 * 2)
    # 固定矿点
    # for i in range(27 * 2):
    #     if i <= 26:
    #         true[i] = 1
    #         pred_21[i] = score_21[mine_random[i][0]][mine_random[i][1]]
    #         pred_39[i] = score_39[mine_random[i][0]][mine_random[i][1]]
    #     else:
    #         true[i] = 0
    #         pred_21[i] = score_21[non_mine_random[i - 27][0]][non_mine_random[i - 27][1]]
    #         pred_39[i] = score_39[non_mine_random[i - 27][0]][non_mine_random[i - 27][1]]
    # 不固定矿点
    for i in range(27 * 2):
        if i <= 26:
            true[i] = 1
            pred_21[i] = score_21[mine_1_random[i][0]][mine_1_random[i][1]]
            pred_39[i] = score_39[mine_1_random[i][0]][mine_1_random[i][1]]
        else:
            true[i] = 0
            pred_21[i] = score_21[non_mine_random[i - 27][0]][non_mine_random[i - 27][1]]
            pred_39[i] = score_39[non_mine_random[i - 27][0]][non_mine_random[i - 27][1]]

    fpr_21, tpr_21, thresholds_21 = roc_curve(true, pred_21)
    max_index = (tpr_21 - fpr_21).tolist().index(max(tpr_21 - fpr_21))
    threshold_21 = thresholds_21[max_index]

    roc_auc_21 = auc(fpr_21, tpr_21)

    plt.plot(fpr_21, tpr_21, 'b--', label='MSE+Phase ROC (area = {0:.3f})'.format(roc_auc_21), lw=2)

    fpr_39, tpr_39, thresholds_39 = roc_curve(true, pred_39)
    max_index = (tpr_39 - fpr_39).tolist().index(max(tpr_39 - fpr_39))
    threshold_39 = thresholds_39[max_index]

    roc_auc_39 = auc(fpr_39, tpr_39)

    plt.plot(fpr_39, tpr_39, 'r--', label='MSE ROC (area = {0:.3f})'.format(roc_auc_39), lw=2)

    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show(block=True)
    return 0


def ROC(is_mask):
    non_mine = np.zeros((102, 82)) + 1
    mine_dh = pd.read_csv('mc/layer/mine_dh.csv')
    mine_27 = np.zeros((102, 82))
    mine_all = pd.read_csv('mc/layer/mine_1.csv')
    mine_1 = np.zeros((102, 82))
    score_39 = np.array(Image.open('img/phase/mask/Au_err_0_True_0.005.tif'))
    # score_39_z = ((score_39 - np.mean(score_39)) / np.var(score_39))
    # arr2raster(score_39_z, 'mc/mineral occurrence.shp', 'mc/result_new/score3_128_5_z.tif')
    score_21 = np.array(Image.open('img/phase/mask/Au_err_2_True_0.008.tif'))

    # mask处理左上角
    mask_arr = np.array(Image.open(r'D:\WGS\DL\fuben\mc\ref\mask.tif'))
    mask_arr = mask_arr / 85

    mine_dh_x = mine_dh['dx'].values
    mine_dh_y = mine_dh['dy'].values
    mine_x = mine_all['dx'].values
    mine_y = mine_all['dy'].values
    x_min = 40476796.08
    y_max = 4170032.922

    for x in range(82):
        for y in range(102):
            x_left = x_min + x * 1000
            y_up = y_max - y * 1000
            if mask_arr[y][x] == 0 and is_mask:
                non_mine[y][x] = -1
                mine_27[y][x] = -1
                mine_1[y][x] = -1
                continue
            for i in range(len(mine_x)):
                if np.sqrt(np.power((mine_x[i] - x_left), 2) + np.power((mine_y[i] - y_up), 2)) < 15000:
                    non_mine[y][x] = 0
            for i in range(len(mine_dh)):
                if (mine_dh_x[i] >= x_left and mine_dh_y[i] <= y_up
                        and mine_dh_x[i] < (x_left + 1000) and mine_dh_y[i] > (y_up - 1000)):
                    mine_27[y][x] = 1
                    mine_1[y][x] = 1

    mine_random = np.argwhere(mine_27 == 1)
    auc_39 = 0
    auc_21 = 0
    times = 100
    for iter in range(times):
        mine_1_index = np.argwhere(mine_1 == 1)
        non_mine_index = np.argwhere(non_mine == 1)

        np.random.shuffle(mine_1_index)
        np.random.shuffle(non_mine_index)

        non_mine_random = non_mine_index[0:27]
        mine_1_random = mine_1_index[0:27]

        true = np.zeros(27 * 2)
        pred_21 = np.zeros(27 * 2)
        pred_39 = np.zeros(27 * 2)

        for i in range(27 * 2):
            if i <= 26:
                true[i] = 1
                pred_21[i] = score_21[mine_1_random[i][0]][mine_1_random[i][1]]
                pred_39[i] = score_39[mine_1_random[i][0]][mine_1_random[i][1]]
            else:
                true[i] = 0
                pred_21[i] = score_21[non_mine_random[i - 27][0]][non_mine_random[i - 27][1]]
                pred_39[i] = score_39[non_mine_random[i - 27][0]][non_mine_random[i - 27][1]]

        fpr_21, tpr_21, thresholds_21 = roc_curve(true, pred_21)
        max_index = (tpr_21 - fpr_21).tolist().index(max(tpr_21 - fpr_21))
        threshold_21 = thresholds_21[max_index]

        roc_auc_21 = auc(fpr_21, tpr_21)

        # plt.plot(fpr_21, tpr_21, 'b--', label='MSE+Phase ROC (area = {0:.3f})'.format(roc_auc_21), lw=2)

        fpr_39, tpr_39, thresholds_39 = roc_curve(true, pred_39)
        max_index = (tpr_39 - fpr_39).tolist().index(max(tpr_39 - fpr_39))
        threshold_39 = thresholds_39[max_index]

        roc_auc_39 = auc(fpr_39, tpr_39)

        # plt.plot(fpr_39, tpr_39, 'r--', label='MSE ROC (area = {0:.3f})'.format(roc_auc_39), lw=2)

        auc_39 = auc_39 + roc_auc_39
        auc_21 = auc_21 + roc_auc_21
        if iter == times - 1:
            auc_39 = auc_39 / times
            auc_21 = auc_21 / times

            plt.plot(fpr_21, tpr_21, 'b--', label='Phase Loss(Au) AUC = {0:.3f}'.format(auc_21), lw=2)
            plt.plot(fpr_39, tpr_39, 'r--', label='MSE AUC(Au) = {0:.3f}'.format(auc_39), lw=2)
            # plt.plot(fpr_21, tpr_21, 'b--', label='MSE+Phase AUC (area = {0:.3f})'.format(auc_21), lw=2)
            # plt.plot(fpr_39, tpr_39, 'r--', label='MSE AUC (area = {0:.3f})'.format(auc_39), lw=2)
            plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.show(block=True)
    print(threshold_39)
    return 0


def CalcAnomalScore(err, mode='all'):
    chosen_index = [0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 20, 21, 22, 24, 28, 30]
    factor_index = [0, 2, 4, 12, 14, 15, 20, 28, 30]
    if mode == 'chosen':
        err = err[chosen_index]
    if mode == 'factor':
        err = err[factor_index]
    score = np.zeros((102, 82))
    for i in range(102):
        for j in range(82):
            score[i][j] = np.sqrt(np.sum(np.power(err[:, i, j], 2)))
    return score


def arr2raster(raster_data, refShpFileName, raster_fn):
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
    # 为了使属性表字段支持中文，请添加下面这句
    gdal.SetConfigOption("SHAPE_ENCODING", "")
    # 注册所有的驱动
    ogr.RegisterAll()
    # 数据格式的驱动
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(refShpFileName, update=1)

    layer0 = ds.GetLayerByIndex(0)

    # 或取到shp的投影坐标系信息
    prosrs = layer0.GetSpatialRef()
    # geoTransform = layer0.GetGeoTransform()

    X_size = raster_data.shape[0]
    Y_size = raster_data.shape[1]
    # val = np.zeros((X_size, Y_size))
    # for col in range(X_size):
    # 	for row in range(Y_size):
    # 		val[row, col] = raster_data[col, row]

    # print(raster_data[498, 200])
    origin = (40476796.08, 4170032.922)
    p_width = 1000
    p_height = -1000

    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(raster_fn, Y_size, X_size, 1, gdal.GDT_Float32)
    out_raster.SetGeoTransform((origin[0], p_width, 0, origin[1], 0, p_height))
    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(raster_data)
    out_raster_SRS = osr.SpatialReference()
    out_raster_SRS.ImportFromEPSG(3857)
    out_raster.SetProjection(prosrs.ExportToWkt())
    out_band.FlushCache()
    return 0