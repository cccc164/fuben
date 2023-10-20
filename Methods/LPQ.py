#!/usr/bin/env Python
# coding=utf-8
from __future__ import division
import numpy as np
import torch
import math
from scipy.signal import convolve2d
import torch.nn.functional as F

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

    # Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis, np.newaxis, :]
    LPQdesc = ((freqResp > 0) * (2 ** inds)).sum(2)

    # Switch format to uint8 if LPQ code np.image is required as output
    if mode == 'im':
        LPQdesc = np.uint8(LPQdesc)

    # Histogram if needed
    if mode == 'nh' or mode == 'h':
        LPQdesc = np.histogram(LPQdesc.flatten(), range(256))[0]

    # Normalize histogram if needed
    if mode=='nh':
        LPQdesc = LPQdesc / LPQdesc.sum()

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
    img = img.detach().unsqueeze(0).unsqueeze(0)
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

    freqResp = torch.dstack([filterResp1.real, filterResp1.imag,
                             filterResp2.real, filterResp2.imag,
                             filterResp3.real, filterResp3.imag,
                             filterResp4.real, filterResp4.imag])
    freqResp[torch.abs(freqResp) < 1e-5] = 0

    # Perform quantization and compute LPQ codewords
    inds = torch.arange(freqResp.shape[2])[None, None, :].to(torch.device("cuda"))
    LPQdesc = torch.sum(((freqResp > 0) * (2 ** inds)), axis=2)
    # Switch format to uint8 if LPQ code np.image is required as output
    if mode == 'im':
        LPQdesc = torch.uint8(LPQdesc)

    # Histogram if needed
    if mode == 'nh' or mode == 'h':
        LPQdesc = torch.histc(LPQdesc.view(-1), 256)

    # Normalize histogram if needed
    if mode == 'nh':
        LPQdesc = LPQdesc / torch.sum(LPQdesc)

    return LPQdesc


def lpq_tensor(img, winSize=3, freqestim=1, mode='nh'):
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
        w0 = torch.ones_like(x)
        w1 = torch.exp(-2 * np.pi * x * STFTalpha * 1j)
        w2 = torch.conj(w1)

        p1d = (0, 0, 1, 1)
        w0 = F.pad(w0, p1d, 'constant', 0)
        w1 = F.pad(w1, p1d, 'constant', 0)
        w2 = F.pad(w2, p1d, 'constant', 0)

    # filterResp1 = convolve2d(convolve2d(img, w0.T, convmode), w1, convmode)
    # filterResp2 = convolve2d(convolve2d(img, w1.T, convmode), w0, convmode)
    # filterResp3 = convolve2d(convolve2d(img, w1.T, convmode), w1, convmode)
    # filterResp4 = convolve2d(convolve2d(img, w1.T, convmode), w2, convmode)
    t1 = F.conv2d(img, w0.t().unsqueeze(0).unsqueeze(0), stride=1, padding='same')
    filterResp1 = F.conv2d(torch.complex(t1, torch.zeros_like(img)), w1.unsqueeze(0).unsqueeze(0), stride=1,
                           padding='same')
    filterResp1 = filterResp1[0][0]

    # t2_real = F.conv2d(img, w1.t().real.unsqueeze(0).unsqueeze(0), stride=1, padding='same')
    # t2_imag = F.conv2d(img, w1.t().imag.unsqueeze(0).unsqueeze(0), stride=1, padding='same')
    t2 = F.conv2d(torch.complex(img, torch.zeros_like(img)), w1.t().unsqueeze(0).unsqueeze(0), stride=1, padding='same')
    filterResp2 = F.conv2d(t2, torch.complex(w0.real.unsqueeze(0).unsqueeze(0), torch.zeros((winSize, winSize))),
                           stride=1, padding='same')
    filterResp2 = filterResp2[0][0]

    t3 = F.conv2d(torch.complex(img, torch.zeros_like(img)), w1.t().unsqueeze(0).unsqueeze(0), stride=1, padding='same')
    filterResp3 = F.conv2d(t3, w1.unsqueeze(0).unsqueeze(0), stride=1, padding='same')
    filterResp3 = filterResp3[0][0]

    t4 = F.conv2d(torch.complex(img, torch.zeros_like(img)), w1.t().unsqueeze(0).unsqueeze(0), stride=1, padding='same')
    filterResp4 = F.conv2d(t4, w2.unsqueeze(0).unsqueeze(0), stride=1, padding='same')
    filterResp4 = filterResp4[0][0]

    freqResp = torch.dstack([filterResp1.real, filterResp1.imag,
                             filterResp2.real, filterResp2.imag,
                             filterResp3.real, filterResp3.imag,
                             filterResp4.real, filterResp4.imag])
    freqResp[torch.abs(freqResp) < 1e-5] = 0

    # Perform quantization and compute LPQ codewords
    inds = torch.arange(freqResp.shape[2])[None, None, :]
    LPQdesc = torch.sum(((freqResp > 0) * (2 ** inds)), axis=2)
    LPQdesc = LPQdesc.cuda()
    # Switch format to uint8 if LPQ code np.image is required as output
    if mode == 'im':
        LPQdesc = torch.uint8(LPQdesc)

    # Histogram if needed
    if mode == 'nh' or mode == 'h':
        LPQdesc = torch.histc(LPQdesc.view(-1), 256)[0]

    # Normalize histogram if needed
    if mode == 'nh':
        LPQdesc = LPQdesc / torch.sum(LPQdesc)

    LPQdesc
    return LPQdesc


def PhaseLoss(img1, img2, winsize):
    LPQ1 = torch.tensor(lpq(img1, winsize))
    LPQ2 = torch.tensor(lpq(img2, winsize))

    # LPQ1 = lpq_tensor_1(img1)
    # LPQ2 = lpq_tensor_1(img2)
    phase_loss = 0
    # for i in range(len(LPQ1)):
    #     phase_loss = phase_loss + np.square(LPQ1[i] - LPQ2[i]) / (LPQ1[i] + LPQ2[i] + 1e-7)
    phase_loss = torch.sum(torch.square(LPQ1 - LPQ2)/(LPQ1 + LPQ2 + 1e-7))
    # phase_loss
    return phase_loss

if __name__ == '__main__':
    a = torch.zeros((100, 100)) + 1
    LPQ_code = lpq_tensor(a.unsqueeze(0).unsqueeze(0))
    # LPQ_code_t = torch.stft(a)
    LPQ_code