import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
from LPQ import PhaseLoss
import sys
from os import path

# 这里相当于把相对路径 .. 添加到pythonpath中
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))



class TestModel(nn.Module):
    def __init__(self, in_channels):
        super(TestModel, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=9, stride=1, padding=4)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=9, stride=1, padding=4)
        self.pool1 = nn.MaxPool2d(2, padding=0)
        self.pool2 = nn.MaxPool2d(2, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.pad = nn.ConstantPad2d(padding=(0, 1, 0, 0), value=0)

        # Decoder
        self.up1 = nn.ConvTranspose2d(32, 32, 2, stride=2, padding=0, output_padding=0)
        self.up2 = nn.ConvTranspose2d(32, 64, 9, stride=2, padding=4)
        self.up3 = nn.ConvTranspose2d(64, 128, 2, stride=2, padding=0)
        self.conv = nn.Conv2d(128, in_channels, 9, stride=1, padding=4)

    def forward(self, image):
        conv1 = self.conv1(image)  # 82x102x5 102x82x5
        relu1 = F.relu(conv1)  # 82x102x128 102x82x128
        pool1 = self.pool1(relu1)  # 41x51x128 51x41x128

        conv2 = self.conv2(pool1)  # 41x51x64 51x41x64
        relu2 = F.relu(conv2)
        pool2 = self.pool2(relu2)  # 21x26x64 26x21x64

        conv3 = self.conv3(pool2)  # 21x26x32 26x21x32
        relu3 = F.relu(conv3)
        pool3 = self.pool3(relu3)

        up1 = self.up1(pool3)  # 21x26x32 26x21x32
        up_relu1 = F.relu(up1)
        up_relu1 = self.pad(up_relu1)

        up2 = self.up2(up_relu1)  # 20x20x32
        up_relu2 = F.relu(up2)

        up3 = self.up3(up_relu2)
        up_relu3 = F.relu(up3)

        logits = self.conv(up_relu3)
        logits = F.sigmoid(logits)
        return logits


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        # print('1')

    # @staticmethod
    def forward(self, pred, truth):
        # pred = pred.cpu().detach().numpy()
        # # truth = truth.cpu().detach().numpy()
        pred_arr = pred.cpu().detach().numpy()
        truth_arr = pred_arr * np.pi
        truth = pred * np.pi
        MSE = nn.MSELoss()
        loss_mse = MSE(pred, truth)
        # loss_mse = MSE(torch.tensor(pred), torch.tensor(truth))
        lose_phase = 0
        lamda = 1e-4
        # lose_phase = torch.sum(pred) + torch.sum(truth)
        for i in range(pred.shape[1]):
            lose_phase = lose_phase + torch.tensor(PhaseLoss(pred_arr[:, i, :, :][0],
                                                             truth_arr[:, i, :, :][0]), requires_grad=True)
        # return torch.tensor(loss_mse + lamda * lose_phase, requires_grad=True)
        return loss_mse + lamda * lose_phase

    # @staticmethod
    # def backward(self):
    #     grad = torch.tensor(1, requires_grad=True)
    #     return grad


def main():
    # input_ilr_tensor, clr_arr = data_Gen()
    input_ilr_tensor = torch.tensor(np.zeros((1, 38, 102, 82)) + 2)
    device = torch.device("cuda")
    for i in range(1):
        net = TestModel(38)
        net = net.float()
        input_ilr_tensor = input_ilr_tensor.float()

        net.to(device)
        input_ilr_tensor = input_ilr_tensor.to(device)

        loss_function = MyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.0001)

        epochs = 1200
        steps = 100
        epoch_list = list()
        loss_epoch = list()
        for epoch in range(epochs):
            running_loss = 0
            for step in range(steps):
                # 梯度置零
                optimizer.zero_grad()
                # 前后传播加优化
                output = net(input_ilr_tensor)
                loss = loss_function(output, input_ilr_tensor)
                loss.backward()
                optimizer.step()

                # 打印统计信息
                # print("running_loss0 ", running_loss)
                running_loss += loss.item()
                # print("running_loss1 ", running_loss)
                # print("steps ", steps)
                if step % steps == steps - 1:
                    # print(running_loss / steps)
                    print("kernel size:{0} epoch:{1} loss:{2}".format(2 * 2 + 1, epoch, running_loss / steps))
                # print("epoch{0} loss:{1}".format(epoch, running_loss / steps))
                if epoch == (epochs - 1) and step == (steps - 1):
                    data_k = output.cpu().detach().numpy()
                    # print("kernel size:{0} origin total:{1} filter total:{2}".format((i + 1) * 2 + 1, data.sum(), data_k.sum()))
                    result = output
                    recons = output.cpu().detach().numpy()
                # torch.save(net.state_dict(), 'par/3_128_9.pt')
                # tensor_to_img(output, "SCAE_result{}_new.tif".format(window_size * 2 + 1))
            # if epoch == 0:
            #     continue
            epoch_list.append(epoch)
            loss_epoch.append(running_loss / steps)
    # plt.plot(epoch_list, loss_epoch)
    # plt.show()
    # recons_reshape = np.zeros((102*82, 38))
    # recons_clr = np.zeros((102*82, 39))
    # recons_res = list()
    # for i in range(38):
    # 	recons_reshape[:, i] = recons[:, i].reshape(-1, 1)[0]
    # for i in range(102*82):
    # 	recons_clr[i] = ilr2clr(recons_reshape[i, :])
    # for i in range(39):
    # 	arr = recons_clr[:, i].reshape((102, 82))
    # 	recons_res.append(arr)
    # recons_res = np.array(recons_res)
    # err = np.subtract(recons_res, clr_arr)
    # err = err.reshape((39, 102, 82))
    # score = CalcAnomalScore(err)
    # arr2raster(score, 'mc/mineral occurrence.shp', 'mc/result_new/score3_128_9.tif')
    torch.cuda.empty_cache()
    return


if __name__ == '__main__':
    # a = np.zeros((100, 100)) + 1
    # print(a)
    main()
