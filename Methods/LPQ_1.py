#!/usr/bin/env Python
# coding=utf-8

from __future__ import division
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import convolve2d
import torch.nn.functional as F
from PIL import Image
# import torchviz


class MaskMSE(nn.Module):
    def __init__(self, mask):
        super(MaskMSE, self).__init__()
        self.is_mask = mask
        if self.is_mask:
            mask_arr = np.array(Image.open(r'D:\WGS\DL\fuben\mc\ref\mask.tif'))
            mask_arr = mask_arr / 85
            mask_tensor = torch.from_numpy(mask_arr)
            mask_tensor = torch.stack([mask_tensor, mask_tensor, mask_tensor, mask_tensor], dim=2)
            self.mask = torch.tensor(mask_tensor, device=torch.device("cuda"))
        else:
            mask_arr = np.ones_like(Image.open(r'D:\WGS\DL\fuben\mc\ref\mask.tif'))
            mask_tensor = torch.from_numpy(mask_arr)
            mask_tensor = torch.stack([mask_tensor, mask_tensor, mask_tensor, mask_tensor], dim=2)
            self.mask = torch.tensor(mask_tensor, device=torch.device("cuda"))

    # @staticmethod
    def forward(self, pred, truth):
        loss = 0
        # for i in range(pred.shape[1]):
        loss = torch.mean(torch.square(torch.subtract(pred, truth)) * self.mask)
        # loss_all = torch.mean(loss)
        return loss


def lpq(img, winSize=7, freqestim=1, mode='nh'):
    rho = 0.90

    STFTalpha = 1 / winSize  # alpha in STFT approaches (for Gaussian derivative alpha=1)
    sigmaS = (winSize - 1) / 4  # Sigma for STFT Gaussian window (applied if freqestim==2)
    sigmaA = 8 / (winSize - 1)  # Sigma for Gaussian derivative quadrature filters (applied if freqestim==3)

    # conv_mode = 'valid' # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).
    conv_mode = 'same'
    img = np.float64(img)   # Convert np.image to double
    r = (winSize - 1) / 2   # Get radius from window size
    x = np.arange(-r, r + 1)[np.newaxis]    # Form spatial coordinates in window

    if freqestim == 1:  # STFT uniform window
        #  Basic STFT filters
        w0 = np.ones_like(x)
        w1 = np.exp(-2*np.pi*x*STFTalpha*1j)
        w2 = np.conj(w1)

    # Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filter
    filterResp1 = convolve2d(convolve2d(img, w0.T, conv_mode), w1, conv_mode)
    filterResp2 = convolve2d(convolve2d(img, w1.T, conv_mode), w0, conv_mode)
    filterResp3 = convolve2d(convolve2d(img, w1.T, conv_mode), w1, conv_mode)
    filterResp4 = convolve2d(convolve2d(img, w1.T, conv_mode), w2, conv_mode)

    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp=np.dstack([filterResp1.real, filterResp1.imag,
                        filterResp2.real, filterResp2.imag,
                        filterResp3.real, filterResp3.imag,
                        filterResp4.real, filterResp4.imag])

    # 直接计算相位
    # freqResp_real = np.dstack([filterResp1.real, filterResp2.real, filterResp3.real, filterResp4.real])
    # freqResp_img = np.dstack([filterResp1.imag, filterResp2.imag, filterResp3.imag, filterResp4.imag])
    #
    # res = map(phase_calc, freqResp_real.reshape((img.shape[0] * img.shape[0], 4)),
    #           freqResp_img.reshape((img.shape[0] * img.shape[0], 4)))
    # phase = np.array(list(res)).reshape((img.shape[0], img.shape[0], 4)) / np.pi

    # Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis, np.newaxis, :]
    LPQdesc = ((freqResp > 0) * (2 ** inds)).sum(2)

    # 转直方图
    # # Switch format to uint8 if LPQ code np.image is required as output
    # if mode == 'im':
    #     LPQdesc = np.uint8(LPQdesc)
    #
    # # Histogram if needed
    # if mode == 'nh' or mode == 'h':
    #     LPQdesc = np.histogram(LPQdesc.flatten(), range(256))[0]
    #
    # # Normalize histogram if needed
    # if mode=='nh':
    #     LPQdesc = LPQdesc / LPQdesc.sum()

    # 返回LPQ编码矩阵
    LPQdesc = LPQdesc / 255

    # 返回LPQ编码矩阵
    # LPQdesc = ((freqResp > 0) * 1) / 8

    # print(LPQdesc)
    # LPQdesc = np.zeros(255)
    return LPQdesc


def conv_complex(img, kernel):
    img_real = img.real
    img_imag = img.imag
    kernel_real = kernel.real
    kernel_imag = kernel.imag
    # img = a + bj kernel = c + dj
    ac = F.conv2d(img_real, kernel_real, stride=1, padding='same')
    bd = F.conv2d(img_imag, kernel_imag, stride=1, padding='same')

    ad = F.conv2d(img_real, kernel_imag, stride=1, padding='same')
    bc = F.conv2d(img_imag, kernel_real, stride=1, padding='same')
    res_real = ac - bd
    res_imag = ad + bc
    res = torch.complex(res_real, res_imag)
    return res


def lpq_tensor_1(img, winSize=3, freqestim=1, mode='nh'):
    img = img.unsqueeze(0).unsqueeze(0)
    img_real = img
    img_imag = torch.zeros_like(img_real)
    img_complex = torch.complex(img_real, img_imag)
    STFTalpha = 1 / winSize  # alpha in STFT approaches (for Gaussian derivative alpha=1)
    sigmaS = (winSize - 1) / 4  # Sigma for STFT Gaussian window (applied if freqestim==2)
    sigmaA = 8 / (winSize - 1)  # Sigma for Gaussian derivative quadrature filters (applied if freqestim==3)

    convmode = 'same'  # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).

    # img = np.float64(img)  # Convert np.image to double
    r = (winSize - 1) / 2  # Get radius from window size
    x = torch.arange(-r, r + 1)[None]  # Form spatial coordinates in window
    x = x * -1

    if freqestim == 1:  # STFT uniform window
        #  Basic STFT filters
        w0 = torch.ones_like(x).to(torch.device("cuda"))
        w1 = torch.exp(-2 * np.pi * x * STFTalpha * 1j).to(torch.device("cuda"))
        w2 = torch.conj(w1).to(torch.device("cuda"))

        p1d = (0, 0, 1, 1)
        w0 = F.pad(w0, p1d, 'constant', 0)
        w0_t = w0.t().unsqueeze(0).unsqueeze(0)
        w1 = F.pad(w1, p1d, 'constant', 0)
        w1_t = w1.t().unsqueeze(0).unsqueeze(0)
        w0 = w0.unsqueeze(0).unsqueeze(0)
        w1 = w1.unsqueeze(0).unsqueeze(0)
        w2 = F.pad(w2, p1d, 'constant', 0)
        w2 = w2.unsqueeze(0).unsqueeze(0)

    w0_real = w0
    w0_imag = torch.zeros_like(w0_real)
    w0_complex = torch.complex(w0_real, w0_imag)

    w0_t_real = w0_t
    w0_t_imag = torch.zeros_like(w0_t_real)
    w0_t_complex = torch.complex(w0_t_real, w0_t_imag)

    filterResp1 = conv_complex(conv_complex(img_complex, w0_t_complex), w1)[0][0]
    filterResp2 = conv_complex(conv_complex(img_complex, w1_t), w0_complex)[0][0]
    filterResp3 = conv_complex(conv_complex(img_complex, w1_t), w1)[0][0]
    filterResp4 = conv_complex(conv_complex(img_complex, w1_t), w2)[0][0]

    # freqResp = torch.dstack([filterResp1.real, filterResp1.imag,
    #                          filterResp2.real, filterResp2.imag,
    #                          filterResp3.real, filterResp3.imag,
    #                          filterResp4.real, filterResp4.imag])
    #
    # # freqResp[torch.abs(freqResp) < 1e-5] = 0
    # freqResp = torch.where(torch.abs(freqResp) < 1e-5, torch.zeros_like(freqResp), freqResp)
    #
    # freqResp = torch.where(freqResp > 0, torch.ones_like(freqResp), torch.zeros_like(freqResp))
    # freqResp[freqResp > 0] = 1
    # freqResp[freqResp <= 0] = 0

    # 直接计算相位
    freqResp_real = torch.dstack([filterResp1.real, filterResp2.real, filterResp3.real, filterResp4.real])
    freqResp_img = torch.dstack([filterResp1.imag, filterResp2.imag, filterResp3.imag, filterResp4.imag])

    phase = torch.arctan(torch.div(freqResp_img, freqResp_real + 1e-7)) / (torch.tensor(np.pi))
    # phase = torch.arctan(imag / (real + 1e-7))
    # freqResp_real_reshape = freqResp_real.reshape((img.shape[2] * img.shape[3], 4))
    # freqResp_img_reshape = freqResp_img.reshape((img.shape[2] * img.shape[3], 4))
    # phase = torch.zeros((img.shape[2] * img.shape[3], 4))
    # for i in range(len(freqResp_real_reshape)):
    #     phase[i] = phase_calc(freqResp_real_reshape[i], freqResp_img_reshape[i])
    # phase = phase.reshape((img.shape[2], img.shape[3], 4))
    # phase = np.array(list(res)).reshape((img.shape[0], img.shape[0], 4)) / np.pi


    # Perform quantization and compute LPQ codewords
    # inds = torch.arange(freqResp.shape[2])[None, None, :].to(torch.device("cuda"))
    # # LPQdesc = torch.sum((torch.tensor((freqResp > 0) * 1., requires_grad=True) * (2 ** inds)), axis=2)
    # LPQdesc = torch.sum((freqResp * (2 ** inds)), axis=2)

    # Switch format to uint8 if LPQ code np.image is required as output
    # if mode == 'im':
    #     LPQdesc = torch.uint8(LPQdesc)
    #
    # # Histogram if needed
    # if mode == 'nh' or mode == 'h':
    #     LPQdesc = torch.histc(LPQdesc.view(-1), 256)
    #
    # # Normalize histogram if needed
    # if mode == 'nh':
    #     LPQdesc = LPQdesc / torch.sum(LPQdesc)

    # LPQdesc = LPQdesc / 255

    return phase



def euclidean(x, y):
    # return np.sqrt(np.sum((x - y)**2) / x.shape[0])
    return np.sqrt(np.sum((x - y) ** 2))

def phase_calc(real, imag):
    return torch.arctan(imag / (real + 1e-7))

def PhaseLoss(img1, img2, winsize, mask):
    LPQ1 = lpq_tensor_1(img1, winsize)
    LPQ2 = lpq_tensor_1(img2, winsize)

    # LPQ1 = torch.tensor(lpq(img1, winsize)).float()
    # LPQ2 = torch.tensor(lpq(img2, winsize)).float()

    # LPQ1 = lpq(img1, winsize)
    # LPQ2 = lpq(img2, winsize)
    # phase_loss = 0

    # 4相位loss
    # res = map(euclidean, LPQ1.reshape((img1.shape[0] * img1.shape[0], 4)), LPQ2.reshape((img1.shape[0] * img1.shape[0], 4)))
    # phase_loss = np.array(list(res)).reshape((img1.shape[0], img1.shape[0])).copy()
    # loss = (phase_loss.sum() / (img1.shape[0] * img1.shape[0]))

    # 8二进制loss
    # res = map(euclidean, LPQ1.reshape((img1.shape[0] * img1.shape[0], 8)),
    #           LPQ2.reshape((img1.shape[0] * img1.shape[0], 8)))
    # phase_loss = np.array(list(res)).reshape((img1.shape[0], img1.shape[0])).copy()
    # loss = (phase_loss.sum() / (img1.shape[0] * img1.shape[0]))

    # phase_loss = torch.sum(torch.square(LPQ1 - LPQ2)/(LPQ1 + LPQ2 + 1e-7))

    loss_f = MaskMSE(mask)
    phase_loss = loss_f(LPQ1, LPQ2)
    # torchviz.make_dot(phase_loss, {"x": img1, "c": phase_loss}).view()
    return phase_loss




if __name__ == '__main__':
    a = torch.rand((5, 5), requires_grad=True, device=torch.device("cuda"))
    b = a + 1
    b = b.to(torch.device("cuda"))
    loss = PhaseLoss(a, b, winsize=3)
    a

