import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
from Methods.data_Gen import *
from Methods.LPQ import PhaseLoss
from Methods.log_normal_data import ilr2clr
from Methods.unet import UNet
from osgeo import gdal
from osgeo import ogr
from osgeo import osr


def CalcAnomalScore(err):
    score = np.zeros((102, 82))
    for i in range(102):
        for j in range(82):
            score[i][j] = np.sqrt(np.sum(np.power(err[:, i, j], 2)))
    # tor = np.percentile(score.flatten(), 95)
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


def ROC_new():
    non_mine = np.zeros((102, 82)) + 1
    mine_dh = pd.read_csv('mc/mine_dh.csv')
    mine_27 = np.zeros((102, 82))
    mine_all = pd.read_csv('mc/mine_1.csv')
    mine_1 = np.zeros((102, 82))
    score_39 = np.array(Image.open('mc/result_new/score3_128_3_500.tif'))
    # score_39_z = ((score_39 - np.mean(score_39)) / np.var(score_39))
    # arr2raster(score_39_z, 'mc/mineral occurrence.shp', 'mc/result_new/score3_128_5_z.tif')
    score_21 = np.array(Image.open('mc/result_new/score3_128_3_500_phase_100_3.tif'))
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


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        # print('1')

    # @staticmethod
    def forward(self, pred, truth):
        # pred = pred.cpu().detach().numpy()
        # # truth = truth.cpu().detach().numpy()

        pred_arr = pred.cpu().detach().numpy()
        truth_arr = truth.cpu().detach().numpy()
        MSE = nn.MSELoss()
        loss_mse = MSE(pred, truth)
        # loss_mse = MSE(torch.tensor(pred), torch.tensor(truth))
        lose_phase = 0
        lamda = 100
        # lose_phase = torch.sum(pred) + torch.sum(truth)
        # for i in range(pred.shape[1]):
        #     lose_phase += torch.tensor(PhaseLoss(pred[:, i, :, :][0], truth[:, i, :, :][0]), device=torch.device("cuda"))
        # res = list(map(PhaseLoss, pred[0], truth[0]))
        sum = [PhaseLoss(x, y) for x, y in zip(pred_arr[0], truth_arr[0])]
        sum
        # a = list(res)
        # lose_phase = torch.tensor(a, device=torch.device("cuda")).sum()
        # lose_phase += torch.tensor(PhaseLoss(pred[:, 0, :, :][0], truth[:, 0, :, :][0]), device=torch.device("cuda"))
            # a = pred_arr[:, i, :, :][0] + pred_arr[:, i, :, :][0]
        # return torch.tensor(loss_mse + lamda * lose_phase, requires_grad=True)
        return loss_mse + lamda * lose_phase
        # return loss_mse


def main():
    input_ilr_tensor, clr_arr = data_Gen()
    chosen_list_len = 39
    analysis_len = 21
    # input_ilr_tensor = input_ilr_tensor[:, 0:21, :, :]
    device = torch.device("cuda")
    for i in range(1):
        net = UNet(in_channels=38, out_channels=38, kernel_size=3, padding=1)
        net = net.float()
        input_ilr_tensor = input_ilr_tensor.float()

        net.to(device)
        input_ilr_tensor = input_ilr_tensor.to(device)

        # loss_function = nn.MSELoss()
        loss_function = MyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        epochs = 500
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
        recons_reshape[:, i] = recons[:, i].reshape(-1, 1)[0]
    for i in range(102 * 82):
        recons_clr[i] = ilr2clr(recons_reshape[i, :])
    for i in range(chosen_list_len):
        arr = recons_clr[:, i].reshape((102, 82))
        recons_res.append(arr)
    recons_res = np.array(recons_res)
    err = np.subtract(recons_res, clr_arr)
    err = err.reshape((chosen_list_len, 102, 82))
    score = CalcAnomalScore(err)
    arr2raster(score, 'mc/mineral occurrence.shp', 'mc/result_new/score3_128_5_500_phase_100_5.tif')
    # arr2raster(score, 'mc/mineral occurrence.shp', 'mc/result_new/score3_128_5_200_Adam_phase_10_3.tif')
    torch.cuda.empty_cache()
    return


if __name__ == '__main__':
    main()
    mask = [[0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]]
    # ROC_new()
