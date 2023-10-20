import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_one(seed, index, input_ilr_tensor, clr_arr, multi):
    setup_seed(seed)
    if not multi:
        input_ele = torch.from_numpy(clr_arr[index])
        input_ele_tensor = input_ele.view((1, 1, 102, 82))
    else:
        input_ele_tensor = input_ilr_tensor
    device = torch.device("cuda")
    chosen_list_len = 39
    for level in range(1):
        # net = ConvAutoencoderOne(3, chosen_list_len - 1)
        net = ConvAutoencoder(chosen_list_len - 1, 3)
        # net = Autoencoder(1)
        # net = ConvAutoencoder(1, 3)
        net = net.float()
        input_ele_tensor = input_ele_tensor.float()
        net.to(device)
        input_ele_tensor = input_ele_tensor.to(device)

        loss_function = nn.MSELoss()
        # loss_function = MyMSE()
        # lamda = 10 ** level
        decay = 0
        lamda = 1e-2
        # loss_function = MyLoss(lamda)
        optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=decay)

        # epochs = ((lamda + 1) * 2 + 1) * 100
        epochs = 150
        steps = 1
        epoch_list = list()
        loss_epoch = list()
        for epoch in range(epochs):
            running_loss = 0
            for step in range(steps):
                # 梯度置零
                optimizer.zero_grad()
                # 前后传播加优化
                output = net(input_ele_tensor)
                loss = loss_function(output, input_ele_tensor)
                loss.backward()
                optimizer.step()

                # 打印统计信息
                running_loss += loss.item()
                if step % steps == steps - 1:
                    print("lamda size:{0} epoch:{1} loss:{2}".format(10 ** level, epoch, running_loss / steps))
                if epoch == (epochs - 1) and step == (steps - 1):
                    torch.save(net.state_dict(), 'par/3_128_9.pt')
            epoch_list.append(epoch)
            loss_epoch.append(running_loss / steps)
        plt.plot(epoch_list, loss_epoch)
        plt.show()
        torch.cuda.empty_cache()


def predict(input_ilr_tensor, clr_arr, multi):
    index = 2
    chosen_list_len = 39
    device = torch.device("cuda")
    # model = ConvAutoencoderOne(3, chosen_list_len - 1)
    model = ConvAutoencoder(chosen_list_len - 1, 3)
    model.load_state_dict(torch.load('par/3_128_9.pt'))
    model.to(device)

    if not multi:
        input_ele = torch.from_numpy(clr_arr[index])
        input_ele_tensor = input_ele.view((1, 1, 102, 82))
    else:
        input_ele_tensor = input_ilr_tensor
    input_ele_tensor = input_ele_tensor.float()
    input_ele_tensor = input_ele_tensor.to(device)

    recons = model(input_ele_tensor)
    recons = recons.cpu().detach().numpy()

    if multi:
        recons_reshape = np.zeros((102 * 82, chosen_list_len - 1))
        recons_clr = np.zeros((102 * 82, chosen_list_len))
        recons_res = list()
        for i in range(chosen_list_len - 1):
            recons_reshape[:, i] = recons[:, i].reshape(-1, 1)[:, 0]
        for i in range(102 * 82):
            recons_clr[i] = ilr2clr(recons_reshape[i, :])
        for i in range(chosen_list_len):
            arr = recons_clr[:, i].reshape((102, 82))
            recons_res.append(arr)
        recons_res = np.array(recons_res)
        err = np.subtract(recons_res, clr_arr)
        err = err.reshape((chosen_list_len, 102, 82))
        score = CalcAnomalScore(err)
        arr2raster(score, 'mc/layer/mineral occurrence.shp', f'img/score.tif')
        recons = recons_res[index]
        err = np.abs(np.subtract(recons, clr_arr[index]))
    if not multi:
        err = np.abs(np.subtract(recons[0][0], clr_arr[index]))
        arr2raster(recons[0][0], 'mc/layer/mineral occurrence.shp', f'img/Au_recons_clr_one.tif')
        arr2raster(clr_arr[index], 'mc/layer/mineral occurrence.shp', f'img/Au_clr.tif')
        arr2raster(err, 'mc/layer/mineral occurrence.shp', f'img/Au_err_one.tif')
    else:
        arr2raster(recons, 'mc/layer/mineral occurrence.shp', f'img/Au_recons_clr.tif')
        arr2raster(clr_arr[index], 'mc/layer/mineral occurrence.shp', f'img/Au_clr.tif')
        arr2raster(err, 'mc/layer/mineral occurrence.shp', f'img/Au_err.tif')



if __name__ == '__main__':
    input_ilr_tensor, clr_arr = data_Gen()
    train_one(seed=20, index=2, input_ilr_tensor=input_ilr_tensor, clr_arr=clr_arr, multi=True)
    predict(input_ilr_tensor, clr_arr, multi=True)