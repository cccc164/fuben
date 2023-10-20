import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from Methods.LPQ_1 import PhaseLoss

class MyMSE(nn.Module):
    def __init__(self):
        super(MyMSE, self).__init__()

    # @staticmethod
    def forward(self, pred, truth):
        MSE = nn.MSELoss()
        # loss_mse = MSE(pred, truth)
        loss = 0
        for i in range(pred.shape[1]):
            loss = loss + MSE(pred[:, i, :, :][0], truth[:, i, :, :][0])
        return loss


class MaskMSE(nn.Module):
    def __init__(self, mask):
        super(MaskMSE, self).__init__()
        self.is_mask = mask
        if self.is_mask:
            mask_arr = np.array(Image.open(r'D:\WGS\DL\fuben\mc\ref\mask.tif'))
            mask_arr = mask_arr / 85
            mask_tensor = torch.from_numpy(mask_arr)
            self.mask = torch.tensor(mask_tensor, device=torch.device("cuda"))
        else:
            mask_arr = np.ones_like(Image.open('mc/研究区域图层文件/mask.tif'))
            mask_tensor = torch.from_numpy(mask_arr)
            self.mask = torch.tensor(mask_tensor, device=torch.device("cuda"))

    # @staticmethod
    def forward(self, pred, truth):
        loss = 0
        for i in range(pred.shape[1]):
            loss = loss + torch.mean(torch.square(torch.subtract(pred[:, i, :, :][0], truth[:, i, :, :][0])) * self.mask)
        # loss_all = torch.mean(loss)
        return loss


class MyLoss(nn.Module):
    def __init__(self, k, k_s, mask):
        super(MyLoss, self).__init__()
        self.k_s = k_s
        self.k = k
        self.is_mask = mask
        if self.is_mask:
            mask_arr = np.array(Image.open(r'D:\WGS\DL\fuben\mc\ref\mask.tif'))
            mask_arr = mask_arr / 85
            mask_tensor = torch.from_numpy(mask_arr)
            self.mask = torch.tensor(mask_tensor, device=torch.device("cuda"))
        else:
            mask_arr = np.ones_like(Image.open(r'D:\WGS\DL\fuben\mc\ref\mask.tif'))
            mask_tensor = torch.from_numpy(mask_arr)
            self.mask = torch.tensor(mask_tensor, device=torch.device("cuda"))
        # print('1')

    # @staticmethod
    def forward(self, pred, truth):
        # pred = pred.cpu().detach().numpy()
        # # truth = truth.cpu().detach().numpy()

        # pred_arr = pred.cpu().detach().numpy()
        # truth_arr = truth.cpu().detach().numpy()
        # truth = pred * np.pi
        if self.is_mask:
            pred = (pred * self.mask).type(torch.FloatTensor)
            pred = pred.to(torch.device("cuda"))
            truth = (truth * self.mask).type(torch.FloatTensor)
            truth = truth.to(torch.device("cuda"))

        MSE = MaskMSE(self.is_mask)
        loss_mse = MSE(pred, truth)
        # loss_mse = MSE(torch.tensor(pred), torch.tensor(truth))
        # lose_phase = torch.tensor(0.0, requires_grad=True)
        lose_phase = 0
        # lamda = 10 ** self.k
        lamda = self.k
        # lose_phase = torch.sum(pred) + torch.sum(truth)
        for i in range(pred.shape[1]):
            lose_phase = lose_phase + PhaseLoss(pred[:, i, :, :][0]
                                                , truth[:, i, :, :][0], self.k_s, self.is_mask)
            # lose_phase = lose_phase + torch.tensor(PhaseLoss(pred[:, i, :, :][0]
            #                                     , truth[:, i, :, :][0], 5), device=torch.device("cuda"))
            # a = pred_arr[:, i, :, :][0] + pred_arr[:, i, :, :][0]
        # return torch.tensor(loss_mse + lamda * lose_phase, requires_grad=True)
        loss_all = loss_mse + lamda * lose_phase
        # loss_all = lamda * lose_phase
        # torchviz.make_dot(loss_all, {"x": pred, "c": loss_all}).view()
        return loss_all
        # return loss_mse

