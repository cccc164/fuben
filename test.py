#!/usr/bin/env Python
# coding=utf-8
import torch
import torch.nn.functional as F
import torchvision.transforms as trans
import numpy as np
import pandas as pd
from PIL import Image
from Methods.LPQ_1 import *


def gen_kernel(kernel_size: int, sigma):
	kernel = np.zeros((kernel_size, kernel_size))
	# if type == 'Gauss':
	# 	sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 1
	center_i = (kernel_size - 1) / 2
	center_j = (kernel_size - 1) / 2
	for i in range(kernel_size):
		for j in range(kernel_size):
			w = np.exp(-(np.square(i - center_i) + np.square(j - center_j)) / (2 * np.square(sigma))) \
				/ (2 * np.pi * np.square(sigma))
			kernel[i][j] = w
	# if type == 'unsymmetric':
	# 	count = 0
	# 	for i in range(kernel_size):
	# 		for j in range(kernel_size):
	# 			count = count + 1
	# 			w = count
	# 			kernel[i][j] = w
	kernel = kernel / np.sum(kernel)
	kernel_tensor = torch.tensor(kernel).unsqueeze(0).unsqueeze(0)
	return kernel_tensor.float()



if __name__ == '__main__':
	# gen_Gauss_kernel(3, 1)
	origin = np.array(Image.open("data/YCZ-filted/simulated_dataset.tif").convert('F'))
	filted_3 = np.array(Image.open("data/YCZ-filted/pre_train3.tif").convert('F'))
	filted_5 = np.array(Image.open("data/YCZ-filted/pre_train5.tif").convert('F'))
	filted_7 = np.array(Image.open("data/YCZ-filted/pre_train7.tif").convert('F'))
	filted_9 = np.array(Image.open("data/YCZ-filted/pre_train9.tif").convert('F'))
	filted_11 = np.array(Image.open("data/YCZ-filted/pre_train11.tif").convert('F'))
	filted_13 = np.array(Image.open("data/YCZ-filted/pre_train13.tif").convert('F'))
	noise = np.random.uniform(-5, 5, (100, 100))
	corrupted = origin + noise
	origin_tesnor = torch.tensor(origin).unsqueeze(0).unsqueeze(0)

	average_w_3 = (torch.ones((3, 3)) / 9).unsqueeze(0).unsqueeze(0)
	average_w_5 = (torch.ones((5, 5)) / 25).unsqueeze(0).unsqueeze(0)
	average_w_7 = (torch.ones((7, 7)) / 49).unsqueeze(0).unsqueeze(0)
	average_w_9 = (torch.ones((9, 9)) / 81).unsqueeze(0).unsqueeze(0)
	average_w_11 = (torch.ones((11, 11)) / 121).unsqueeze(0).unsqueeze(0)
	average_w_13 = (torch.ones((13, 13)) / 169).unsqueeze(0).unsqueeze(0)
	average_3 = F.conv2d(origin_tesnor, average_w_3, stride=1, padding='same')
	average_3 = average_3[0][0]
	average_5 = F.conv2d(origin_tesnor, average_w_5, stride=1, padding='same')
	average_5 = average_5[0][0]
	average_7 = F.conv2d(origin_tesnor, average_w_7, stride=1, padding='same')
	average_7 = average_7[0][0]
	average_9 = F.conv2d(origin_tesnor, average_w_9, stride=1, padding='same')
	average_9 = average_9[0][0]
	average_11 = F.conv2d(origin_tesnor, average_w_11, stride=1, padding='same')
	average_11 = average_11[0][0]
	average_13 = F.conv2d(origin_tesnor, average_w_13, stride=1, padding='same')
	average_13 = average_13[0][0]

	# Gauss_w_3 = torch.Tensor(
	# 	np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])).unsqueeze(0).unsqueeze(0)
	# Gauss_w_3 = gen_kernel(3, type='Gauss')
	# Gauss_w_5 = gen_kernel(5, type='Gauss')
	# Gauss_w_7 = gen_kernel(7, type='Gauss')
	# Gauss_w_9 = gen_kernel(9, type='Gauss')
	# Gauss_w_11 = gen_kernel(11, type='Gauss')
	# Gauss_w_13 = gen_kernel(13, type='Gauss')

	Gauss_w_3 = gen_kernel(3, sigma=1)
	Gauss_w_5 = gen_kernel(5, sigma=1)
	Gauss_w_7 = gen_kernel(7, sigma=1)
	Gauss_w_9 = gen_kernel(9, sigma=1)
	Gauss_w_11 = gen_kernel(11, sigma=1)
	Gauss_w_13 = gen_kernel(13, sigma=1)

	# unsymmetric_w_3 = gen_kernel(3, type='unsymmetric')
	# unsymmetric_w_5 = gen_kernel(5, type='unsymmetric')
	# unsymmetric_w_7 = gen_kernel(7, type='unsymmetric')

	# unsymmetric_w = torch.Tensor(
	# 	np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) / 45).unsqueeze(0).unsqueeze(0)

	# Gauss_w_5 = torch.Tensor(
	# 	np.array([[1, 4, 7, 4, 1],
	# 			  [4, 16, 26, 16, 4],
	# 			  [7, 26, 41, 26, 7],
	# 			  [4, 16, 26, 16, 4],
	# 			  [1, 4, 7, 4, 1]])).unsqueeze(0).unsqueeze(0)
	# Gauss_w_5 = Gauss_w_5 / 273
	Gauss_3 = F.conv2d(origin_tesnor, Gauss_w_3, stride=1, padding='same')
	Gauss_3 = np.array(Gauss_3)[0][0]

	Gauss_5 = F.conv2d(origin_tesnor, Gauss_w_5, stride=1, padding='same')
	Gauss_5 = np.array(Gauss_5)[0][0]

	Gauss_7 = F.conv2d(origin_tesnor, Gauss_w_7, stride=1, padding='same')
	Gauss_7 = np.array(Gauss_7)[0][0]

	Gauss_9 = F.conv2d(origin_tesnor, Gauss_w_9, stride=1, padding='same')
	Gauss_9 = np.array(Gauss_9)[0][0]

	Gauss_11 = F.conv2d(origin_tesnor, Gauss_w_11, stride=1, padding='same')
	Gauss_11 = np.array(Gauss_11)[0][0]

	Gauss_13 = F.conv2d(origin_tesnor, Gauss_w_13, stride=1, padding='same')
	Gauss_13 = np.array(Gauss_13)[0][0]

	# unsymmetric_3 = F.conv2d(origin_tesnor, unsymmetric_w_3, stride=1, padding='same')
	# unsymmetric_3 = np.array(unsymmetric_3)[0][0]
	#
	# unsymmetric_5 = F.conv2d(origin_tesnor, unsymmetric_w_5, stride=1, padding='same')
	# unsymmetric_5 = np.array(unsymmetric_5)[0][0]
	#
	# unsymmetric_7 = F.conv2d(origin_tesnor, unsymmetric_w_7, stride=1, padding='same')
	# unsymmetric_7 = np.array(unsymmetric_7)[0][0]

	# valid_loss = PhaseLoss(Gauss_3, average_3, 3)
	# valid_origin = origin[1:99, 1:99]
	# valid_filted_3 = filted_3[1:99, 1:99]
	# valid_loss_ycz = PhaseLoss(valid_origin, valid_filted_3, 3)
	# valid_loss_gauss = PhaseLoss(valid_origin, Gauss_3, 3)
	# valid_loss_aver = PhaseLoss(valid_origin, average_3, 3)

	loss_3 = PhaseLoss(origin, filted_3, 3)
	loss_5 = PhaseLoss(origin, filted_5, 5)
	loss_7 = PhaseLoss(origin, filted_7, 7)
	loss_9 = PhaseLoss(origin, filted_9, 9)
	loss_11 = PhaseLoss(origin, filted_11, 11)
	loss_13 = PhaseLoss(origin, filted_13, 13)

	corrupted_loss_3 = PhaseLoss(origin, Gauss_3, 3)
	corrupted_loss_5 = PhaseLoss(origin, Gauss_5, 5)
	corrupted_loss_7 = PhaseLoss(origin, Gauss_7, 7)
	corrupted_loss_9 = PhaseLoss(origin, Gauss_9, 9)
	corrupted_loss_11 = PhaseLoss(origin, Gauss_11, 11)
	corrupted_loss_13 = PhaseLoss(origin, Gauss_13, 13)

	# corrupted_loss_3 = PhaseLoss(origin, Gauss_3, 3)
	# corrupted_loss_5 = PhaseLoss(origin, Gauss_5, 5)
	# corrupted_loss_7 = PhaseLoss(origin, Gauss_7, 7)
	# corrupted_loss_9 = PhaseLoss(origin, Gauss_9, 9)
	# corrupted_loss_11 = PhaseLoss(origin, Gauss_11, 11)
	# corrupted_loss_13 = PhaseLoss(origin, Gauss_13, 13)

	aver_loss_3 = PhaseLoss(origin, average_3, 3)
	aver_loss_5 = PhaseLoss(origin, average_5, 5)
	aver_loss_7 = PhaseLoss(origin, average_7, 7)
	aver_loss_9 = PhaseLoss(origin, average_9, 9)
	aver_loss_11 = PhaseLoss(origin, average_11, 11)
	aver_loss_13 = PhaseLoss(origin, average_13, 13)

	# unsymmetric_loss_3 = PhaseLoss(origin, unsymmetric_3, 3)
	# unsymmetric_loss_5 = PhaseLoss(origin, unsymmetric_5, 5)
	# unsymmetric_loss_7 = PhaseLoss(origin, unsymmetric_7, 7)

	noise_loss = PhaseLoss(origin, corrupted, 3)
	corrupted_loss_3
