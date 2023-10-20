#!/usr/bin/env Python
# coding=utf-8
import numpy as np

from Methods.YCZ_filtering import *
# 处理闭合效应
from Methods.log_normal_data import *
from Methods.YCZ_spatial_interpolation_new import *
from os import path
from sklearn import preprocessing
import pandas as pd
import torch
import random
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
# from pykrige.ok import OrdinaryKriging

from PIL import Image


def solve_data(process_data, process_loc):
    shape = process_data.shape
    solved_data = list()
    solved_loc = list()
    sum_num = 0
    for i in range(shape[0]):
        new_data = list()
        new_loc = list()
        if not (np.isnan(process_data[i])):
            new_data.append(process_data[i])
            new_loc.append([process_loc[i][0], process_loc[i][1]])
            sum_num += 1
            solved_data.append(process_data[i])
            solved_loc.append([process_loc[i][0], process_loc[i][1]])
    print(sum_num)

    return solved_data, solved_loc


def ArcGIS_data():
    chosen_list = ['Ag', 'Au', 'Cu', 'Sb', 'Fe2O3']
    res = dict()
    for key in chosen_list:
        image_path = "input"+key+".tif"
        # 读取图片
        image = Image.open(image_path)
        # 输出图片维度
        data = np.array(image)[:, :82]
        data = data.reshape((82, 102))
        res[key] = data
    image_path = "input.tif"
    # # 读取图片
    # image = Image.open(image_path)
    # # 输出图片维度
    # data = np.array(image)[:81, :]
    return res


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


def data_Gen_one(ele: str):
    data_dir = "D:/WGS/WGS_holiday/文献/实验/数据及代码/Phase-Ex/data"

    gc_data = path.join(data_dir, "../SGSIM_39_ori.csv")
    gc = pd.read_csv(gc_data)
    ele_col = gc[ele].values
    row = gc['row'].values
    col = gc['col'].values
    ele_arr = np.zeros((102, 82))
    for iter in range(82 * 102):
        ele_arr[row[iter]][col[iter]] = ele_col[iter]

    ele_arr_nor = (ele_arr - np.min(ele_arr)) / (np.max(ele_arr) - np.min(ele_arr))
    ele_arr_nor
    return 0


def data_Gen(mask: bool):
    data_dir = "D:/WGS/WGS_holiday/文献/实验/数据及代码/Phase-Ex/data"

    gc_data = path.join(data_dir, "../SGSIM_39_ori.csv")
    # 取出地化数据
    element_list = ['Ag', 'As_', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Cd', 'Co', 'Cr',
                    'Cu', 'F', 'Hg', 'La',
                    'Li', 'Mn', 'Mo', 'Nb', 'Ni', 'P',
                    'Pb', 'Sb', 'Sn', 'Sr', 'Th', 'Ti', 'U', 'V', 'W', 'Y',
                    'Zn', 'Zr', 'Al2O3', 'CaO', 'Fe2O3', 'K2O', 'MgO', 'Na2O', 'SiO2']
    # 21变量顺序
    # chosen_list = ['Ag', 'As_', 'Au', 'Ba', 'Bi', 'Cd', 'Co', 'Cr',
    #                'Cu', 'F', 'Hg', 'Li', 'Mn', 'Mo', 'Ni', 'W',
    #                'Pb', 'Sb', 'Sn', 'Th', 'Zn', 'B', 'Be', 'La', 'Nb', 'P', 'Sr', 'Ti', 'U', 'V',
    #                'Y', 'Zr', 'Al2O3', 'CaO', 'Fe2O3', 'K2O', 'MgO', 'Na2O', 'SiO2']
    # factor element
    factor_list = ['Ag', 'Au', 'Ba', 'Hg', 'Li', 'Mn', 'Pb', 'W', 'Zn']
    # 正常变量顺序
    chosen_list = ['Ag', 'As_', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Cd', 'Co', 'Cr',
                    'Cu', 'F', 'Hg', 'La',
                    'Li', 'Mn', 'Mo', 'Nb', 'Ni', 'P',
                    'Pb', 'Sb', 'Sn', 'Sr', 'Th', 'Ti', 'U', 'V', 'W', 'Y',
                    'Zn', 'Zr', 'Al2O3', 'CaO', 'Fe2O3', 'K2O', 'MgO', 'Na2O', 'SiO2']
    kun_list = ['Ag', 'As_', 'Au', 'Bi', 'Cu']


    gc = pd.read_csv(gc_data)

    # 处理闭合效应
    dd = Data_deal(element_list)
    ilr_data, cor, clr_data = dd.pre_process_data(gc_data, chosen_list)

    ilr_list = list()
    clr_list = list()
    i = 0
    for i in range(len(chosen_list) - 1):
        ilr_value = np.array(ilr_data)[i, :]
        clr_value = clr_data[:, i]
        row = gc['row'].values
        col = gc['col'].values
        ilr_arr = np.zeros((102, 82))
        clr_arr = np.zeros((102, 82))
        for iter in range(82 * 102):
            ilr_arr[row[iter]][col[iter]] = ilr_value[iter]
            clr_arr[row[iter]][col[iter]] = clr_value[iter]
        ilr_list.append(ilr_arr)
        clr_list.append(clr_arr)
    # clr 多一维
    clr_value = clr_data[:, len(chosen_list) - 1]
    row = gc['row'].values
    col = gc['col'].values
    clr_arr = np.zeros((102, 82))
    for iter in range(82 * 102):
        clr_arr[row[iter]][col[iter]] = clr_value[iter]
    clr_list.append(clr_arr)

    ilr_marr = np.array(ilr_list)
    clr_marr = np.array(clr_list)

    for i in range(39):
        arr2raster(clr_marr[i, :, :], 'mc/layer/mineral occurrence.shp', f'data/element/{chosen_list[i]}.tif')
    # 存金元素
    # Au_clr = clr_marr[2]
    # arr2raster(Au_clr, 'mc/layer/mineral occurrence.shp', f'img/Au_clr.tif')

    # for key in chosen_list:
    #     temp = ilr_list[key].reshape((102, 82))
    #     ilr_marr.append(temp)
    # ilr_marr = np.array(ilr_marr)
    ilr_marr = torch.from_numpy(ilr_marr)
    ilr_marr = ilr_marr.view([1, len(chosen_list) - 1, 102, 82])

    # if mask:
    # mask处理左上角
    mask_arr = np.array(Image.open('mc/ref/mask.tif'))
    mask_arr = mask_arr / 85
    mask_tensor = torch.from_numpy(mask_arr)
    mask_tensor = mask_tensor.view([1, 1, 102, 82])
    ilr_marr_mask = ilr_marr.mul(mask_tensor)
    clr_marr_mask = np.multiply(clr_marr, mask_arr)
    if mask:
        return ilr_marr_mask, clr_marr_mask, mask_arr
    else:
        return ilr_marr, clr_marr, mask_arr


def data_Gen_5(mask: bool):
    data_dir = "D:/WGS/WGS_holiday/文献/实验/数据及代码/Phase-Ex/data"

    gc_data = path.join(data_dir, "../SGSIM_39_ori.csv")
    # 取出地化数据
    element_list = ['Ag', 'As_', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Cd', 'Co', 'Cr',
                    'Cu', 'F', 'Hg', 'La',
                    'Li', 'Mn', 'Mo', 'Nb', 'Ni', 'P',
                    'Pb', 'Sb', 'Sn', 'Sr', 'Th', 'Ti', 'U', 'V', 'W', 'Y',
                    'Zn', 'Zr', 'Al2O3', 'CaO', 'Fe2O3', 'K2O', 'MgO', 'Na2O', 'SiO2']
    chosen_list = ['Ag', 'As_', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Cd', 'Co', 'Cr',
                    'Cu', 'F', 'Hg', 'La',
                    'Li', 'Mn', 'Mo', 'Nb', 'Ni', 'P',
                    'Pb', 'Sb', 'Sn', 'Sr', 'Th', 'Ti', 'U', 'V', 'W', 'Y',
                    'Zn', 'Zr', 'Al2O3', 'CaO', 'Fe2O3', 'K2O', 'MgO', 'Na2O', 'SiO2']
    kun_list = ['Ag', 'As_', 'Au', 'Bi', 'Cu']


    df = pd.read_csv(gc_data)
    loc = df[['col', 'row']]
    gc_data = df[kun_list]
    # 处理闭合效应
    # dd = Data_deal(element_list)
    # ilr_data, cor, clr_data = dd.pre_process_data(gc_data, chosen_list)

    #归一化
    data_nor = np.zeros_like(gc_data.values)
    for i in range(5):
        data_nor[:, i] = (gc_data.values[:, i] - np.min(gc_data.values[:, i])) \
                         / (np.max(gc_data.values[:, i]) - np.min(gc_data.values[:, i]))
    # data_nor = mix_max_scaler.fit_transform(gc_data.values)
    data_nor

    input_list = list()
    ilr_list = list()
    clr_list = list()
    i = 0
    for i in range(len(kun_list)):
        nor_value = np.array(data_nor)[:, i]
        row = loc['row'].values
        col = loc['col'].values
        input_arr = np.zeros((102, 82))
        for iter in range(82 * 102):
            input_arr[row[iter]][col[iter]] = nor_value[iter]
        input_list.append(input_arr)
    input_marr = np.array(input_list)
    ilr_marr = torch.from_numpy(input_marr)
    ilr_marr = ilr_marr.view([1, len(kun_list), 102, 82])

    # if mask:
    # mask处理左上角
    mask_arr = np.array(Image.open(r'D:\WGS\DL\fuben\mc\ref\mask.tif'))
    mask_arr = mask_arr / 85
    mask_tensor = torch.from_numpy(mask_arr)
    mask_tensor = mask_tensor.view([1, 1, 102, 82])
    input_marr_mask = ilr_marr.mul(mask_tensor)
    if mask:
        return input_marr_mask, input_marr_mask, mask_arr
    else:
        return ilr_marr, input_marr, mask_arr


if __name__ == '__main__':
    data_Gen_5(True)
