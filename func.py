import numpy as np
import pandas as pd
import torch


def main():
    input_ilr_tensor, clr_arr = data_Gen()
    chosen_list_len = 39
    analysis_len = 21
    # input_ilr_tensor = input_ilr_tensor[:, 0:21, :, :]
    device = torch.device("cuda")
    for level in range(1):
        # net = Autoencoder(chosen_list_len - 1)
        net = Autoencoder(chosen_list_len - 1)
        net = net.float()
        input_ilr_tensor = input_ilr_tensor.float()

        net.to(device)
        input_ilr_tensor = input_ilr_tensor.to(device)

        # loss_function = nn.MSELoss()
        loss_function = MyMSE()
        lamda = 10 ** level
        decay = 0
        # lamda = 0
        # loss_function = MyLoss(lamda)
        optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=decay)

        # epochs = ((lamda + 1) * 2 + 1) * 100
        epochs = 300
        steps = 1
        epoch_list = list()
        loss_epoch = list()
        recons = 0
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
                    print("lamda size:{0} epoch:{1} loss:{2}".format(10**level, epoch, running_loss / steps))
                # print("epoch{0} loss:{1}".format(epoch, running_loss / steps))
                if epoch == (epochs - 1) and step == (steps - 1):
                    data_k = output.cpu().detach().numpy()
                    # print("kernel size:{0} origin total:{1} filter total:{2}".format((i + 1) * 2 + 1, data.sum(), data_k.sum()))
                    result = output
                    recons = output.cpu().detach().numpy()
                    torch.save(net.state_dict(), 'par/3_128_9.pt')
                # tensor_to_img(output, "SCAE_result{}_new.tif".format(window_size * 2 + 1))
            # if epoch == 0:
            #     continue
            epoch_list.append(epoch)
            loss_epoch.append(running_loss / steps)
        plt.plot(epoch_list, loss_epoch)
        plt.show()
        # 以下为38通道ilr转clr再计算err代码
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
        # 单元素结果
        recons_res = np.array(recons_res)
        err = np.abs(np.subtract(recons_res[2], clr_arr[2]))
        arr2raster(recons_res[2], 'mc/layer/mineral occurrence.shp', f'img/Au_recons_clr.tif')
        arr2raster(clr_arr[2], 'mc/layer/mineral occurrence.shp', f'img/Au_clr.tif')
        arr2raster(err, 'mc/layer/mineral occurrence.shp', f'img/Au_err.tif')
        # err = err.reshape((chosen_list_len, 102, 82))
        # score = CalcAnomalScore(err)

        # arr2raster(score, 'mc/mineral occurrence.shp', f'mc/kernel_size/score3_128_5_{epochs}_phase_{10**lamda}_5.tif')
        # arr2raster(score, 'mc/layer/mineral occurrence.shp', f'mc/nonphase/score3_128_3_{epochs}_{decay}.tif')
        # arr2raster(score, 'mc/layer/mineral occurrence.shp', f'mc/phase/score3_128_3_{epochs}_newphase_{lamda}_7.tif')
        torch.cuda.empty_cache()

    return


def OneEle(index):
    input_ilr_tensor, clr_arr = data_Gen()
    input_ele = torch.from_numpy(clr_arr[index])
    input_ele_tensor = input_ele.view((1, 1, 102, 82))
    chosen_list_len = 39
    device = torch.device("cuda")
    for level in range(1):
        net = ConvAutoencoderOne(31)
        # net = Autoencoder(1)
        # net = ConvAutoencoder(1, 3)
        net = net.float()
        input_ele_tensor = input_ele_tensor.float()

        net.to(device)
        input_ele_tensor = input_ele_tensor.to(device)

        # loss_function = nn.MSELoss()
        loss_function = MyMSE()
        lamda = 10 ** level
        decay = 0
        # lamda = 0
        # loss_function = MyLoss(lamda)
        optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=decay)

        # epochs = ((lamda + 1) * 2 + 1) * 100
        epochs = 300
        steps = 1
        epoch_list = list()
        loss_epoch = list()
        recons = 0
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
                # print("running_loss0 ", running_loss)
                running_loss += loss.item()
                # print("running_loss1 ", running_loss)
                # print("steps ", steps)
                if step % steps == steps - 1:
                    # print(running_loss / steps)
                    print("lamda size:{0} epoch:{1} loss:{2}".format(10 ** level, epoch, running_loss / steps))
                # print("epoch{0} loss:{1}".format(epoch, running_loss / steps))
                if epoch == (epochs - 1) and step == (steps - 1):
                    data_k = output.cpu().detach().numpy()
                    # print("kernel size:{0} origin total:{1} filter total:{2}".format((i + 1) * 2 + 1, data.sum(), data_k.sum()))
                    result = output
                    # recons = output.cpu().detach().numpy()
                    torch.save(net.state_dict(), 'par/3_128_9.pt')
                # tensor_to_img(output, "SCAE_result{}_new.tif".format(window_size * 2 + 1))
            # if epoch == 0:
            #     continue
            epoch_list.append(epoch)
            loss_epoch.append(running_loss / steps)
        plt.plot(epoch_list, loss_epoch)
        plt.show()
        # 以下为38通道ilr转clr再计算err代码
        # recons_reshape = np.zeros((102 * 82, chosen_list_len - 1))
        # recons_clr = np.zeros((102 * 82, chosen_list_len))
        # recons_res = list()
        # for i in range(chosen_list_len - 1):
        #     recons_reshape[:, i] = recons[:, i].reshape(-1, 1)[:, 0]
        # for i in range(102 * 82):
        #     recons_clr[i] = ilr2clr(recons_reshape[i, :])
        # for i in range(chosen_list_len):
        #     arr = recons_clr[:, i].reshape((102, 82))
        #     recons_res.append(arr)
        # recons_res = np.array(recons_res)
        # err = np.subtract(recons_res, clr_arr)
        # err = err.reshape((chosen_list_len, 102, 82))
        # score = CalcAnomalScore(err)
        # arr2raster(score, 'mc/layer/mineral occurrence.shp', f'img/score.tif')
        # 单元素结果
        # recons_res = np.array(recons_res)

        # err = np.abs(np.subtract(recons[0][0], clr_arr[index]))
        # recons_nor = (recons[0][0] - np.min(clr_arr[index])) / (np.max(clr_arr[index]) - np.min(clr_arr[index]))
        # clr_nor = (clr_arr[index] - np.min(clr_arr[index])) / (np.max(clr_arr[index]) - np.min(clr_arr[index]))
        # err = (err - np.min(err)) / (np.max(err) - np.min(err))
        # arr2raster(recons_nor, 'mc/layer/mineral occurrence.shp', f'img/Au_recons_clr_one.tif')
        # arr2raster(clr_nor, 'mc/layer/mineral occurrence.shp', f'img/Au_clr.tif')
        # arr2raster(err, 'mc/layer/mineral occurrence.shp', f'img/Au_err_one.tif')
        model = ConvAutoencoderOne(31)
        model.load_state_dict(torch.load('par/3_128_9.pt'))
        model.to(device)
        recons = model(input_ele_tensor)
        recons = recons.cpu().detach().numpy()
        err = np.abs(np.subtract(recons[0][0], clr_arr[index]))
        arr2raster(recons[0][0], 'mc/layer/mineral occurrence.shp', f'img/Au_recons_clr_one.tif')
        arr2raster(clr_arr[index], 'mc/layer/mineral occurrence.shp', f'img/Au_clr.tif')
        arr2raster(err, 'mc/layer/mineral occurrence.shp', f'img/Au_err_one.tif')

        # err = err.reshape((chosen_list_len, 102, 82))
        # score = CalcAnomalScore(err)

        # arr2raster(score, 'mc/mineral occurrence.shp', f'mc/kernel_size/score3_128_5_{epochs}_phase_{10**lamda}_5.tif')
        # arr2raster(score, 'mc/layer/mineral occurrence.shp', f'mc/nonphase/score3_128_3_{epochs}_{decay}.tif')
        # arr2raster(score, 'mc/layer/mineral occurrence.shp', f'mc/phase/score3_128_3_{epochs}_newphase_{lamda}_7.tif')
        torch.cuda.empty_cache()




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
    return solved_data, solved_loc


def ycz_one(clr_arr):
    # data = pd.read_csv('SGSIM_39_ori.csv')
    # Au_ori = data['Au'].values
    Au_clr = clr_arr[2]
    origin = (40476796.08, 4170032.922)
    p_width = 1000
    p_height = -1000
    ele = np.reshape(Au_clr, (102 * 82, 1))
    loc = np.zeros((102, 82, 2))
    for row in range(102):
        for col in range(82):
            loc[row][col][0] = origin[0] + p_width * col
            loc[row][col][1] = origin[1] + p_height * row
    loc = np.reshape(loc, (102 * 82, 2))
    element, loc = solve_data(ele, loc)
    YCZ = YCZ_filter(element, loc)
    # repeat_times, X, Y = YCZ.plot_random_fuc()
    repeat_times = 20
    filtered_value = YCZ.solve_filtered_value(repeat_times)
    filtered_value = np.reshape(filtered_value, (102, 82))

    err = np.abs(np.subtract(Au_clr, filtered_value))
    arr2raster(filtered_value, 'mc/layer/mineral occurrence.shp', f'img/ycz/Au_filtered_clr_ycz{repeat_times}.tif')
    # arr2raster(clr_arr[index], 'mc/layer/mineral occurrence.shp', f'img/Au_clr.tif')
    arr2raster(err, 'mc/layer/mineral occurrence.shp', f'img/ycz/Au_err_ycz{repeat_times}.tif')


def adjust_par():
    non_mine = np.zeros((102, 82)) + 1
    mine_dh = pd.read_csv('mc/mine_dh.csv')
    mine_27 = np.zeros((102, 82))
    mine_all = pd.read_csv('mc/mine_1.csv')
    mine_1 = np.zeros((102, 82))
    auc_39_dict = dict()
    for epoch in range(2):
        score_39 = np.array(Image.open(f'mc/epoch/score3_128_3_{((epoch + 1) * 2 + 1) * 100}_phase_10_3.tif'))
        # score_39_z = ((score_39 - np.mean(score_39)) / np.var(score_39))
        # arr2raster(score_39_z, 'mc/mineral occurrence.shp', 'mc/result_new/score3_128_5_z.tif')
        # score_21 = np.array(Image.open('mc/kernel_size/score3_128_7_300_phase_1_7.tif'))
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
                    # pred_21[i] = score_21[mine_1_random[i][0]][mine_1_random[i][1]]
                    pred_39[i] = score_39[mine_1_random[i][0]][mine_1_random[i][1]]
                else:
                    true[i] = 0
                    # pred_21[i] = score_21[non_mine_random[i - 27][0]][non_mine_random[i - 27][1]]
                    pred_39[i] = score_39[non_mine_random[i - 27][0]][non_mine_random[i - 27][1]]

            # fpr_21, tpr_21, thresholds_21 = roc_curve(true, pred_21)
            # max_index = (tpr_21 - fpr_21).tolist().index(max(tpr_21 - fpr_21))
            # threshold_21 = thresholds_21[max_index]
            #
            # roc_auc_21 = auc(fpr_21, tpr_21)

            # plt.plot(fpr_21, tpr_21, 'b--', label='MSE+Phase ROC (area = {0:.3f})'.format(roc_auc_21), lw=2)

            fpr_39, tpr_39, thresholds_39 = roc_curve(true, pred_39)
            max_index = (tpr_39 - fpr_39).tolist().index(max(tpr_39 - fpr_39))
            threshold_39 = thresholds_39[max_index]

            roc_auc_39 = auc(fpr_39, tpr_39)

            # plt.plot(fpr_39, tpr_39, 'r--', label='MSE ROC (area = {0:.3f})'.format(roc_auc_39), lw=2)

            auc_39 = auc_39 + roc_auc_39
            # auc_21 = auc_21 + roc_auc_21
            if iter == times - 1:
                auc_39_dict[epoch] = auc_39 / times
                # auc_21 = auc_21 / times

                # plt.plot(fpr_21, tpr_21, 'b--', label='MSE+Phase ROC (area = {0:.3f})'.format(auc_21), lw=2)
                # plt.plot(fpr_39, tpr_39, 'r--', label='MSE ROC (area = {0:.3f})'.format(auc_39), lw=2)
                # plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
                # plt.ylim([-0.05, 1.05])
                # plt.xlabel('False Positive Rate')
                # plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
                # plt.title('ROC Curve')
                # plt.legend(loc="lower right")
                # plt.show(block=True)
    auc_39_dict
    return 0

def getArr(filename):
    data = pd.read_csv(filename, delimiter=',')
    test = 0
    # chosen_list = ['Ag', 'As_', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Cd', 'Co', 'Cr',
    #                 'Cu', 'F', 'Hg', 'La',
    #                 'Li', 'Mn', 'Mo', 'Nb', 'Ni', 'P',
    #                 'Pb', 'Sb', 'Sn', 'Sr', 'Th', 'Ti', 'U', 'V', 'W', 'Y',
    #                 'Zn', 'Zr', 'Al2O3', 'CaO', 'Fe2O3', 'K2O', 'MgO', 'Na2O', 'SiO2']

    chosen_list = ['Ag', 'As_', 'Au', 'Ba', 'Bi', 'Cd', 'Co', 'Cr',
                   'Cu', 'F', 'Hg', 'Li', 'Mn', 'Mo', 'Ni', 'W',
                   'Pb', 'Sb', 'Sn', 'Th', 'Zn']
    res = dict()
    for key in chosen_list:
        value = data[key].values
        row = data['row'].values
        col = data['col'].values
        arr = np.zeros((102, 82))
        for iter in range(82 * 102):
            arr[row[iter]][col[iter]] = value[iter]
        res[key] = arr

    MArr = list()
    for key in chosen_list:
        temp = res[key].reshape((102, 82))
        MArr.append(temp)
    MArr = np.array(MArr)
    MArr = torch.from_numpy(MArr)
    MArr = MArr.view([1, 39, 102, 82])
    return MArr


def array2tensor(data):
    MArr = list()
    for key in data.keys():
        MArr.append(data[key])
    MArr = np.array(MArr)
    MArr = torch.from_numpy(MArr)
    MArr = MArr.view([1, 39, 82, 102])
    return MArr