from Methods.model import *
import os
import argparse
from Methods.calculateROC import CalcAnomalScore, arr2raster, ROC
from Methods.loss import MyLoss
from Methods.data_Gen import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def train_one(index, input_ilr_tensor, clr_arr, multi):
    if not multi:
        input_ele = torch.from_numpy(clr_arr[index])
        input_ele_tensor = input_ele.view((1, 1, 102, 82))
    else:
        input_ele_tensor = input_ilr_tensor
    device = torch.device("cuda")
    chosen_list_len = 39
    for level in range(1):
        # net = ConvAutoencoderOne(3, chosen_list_len - 1)
        # net = ConvAutoencoder(3, chosen_list_len - 1)
        net = ConvAutoencoderSample(args.k_s, args.in_len, base=args.con_c)
        # net = Autoencoder(chosen_list_len - 1)
        # net = ConvAutoencoder(1, 3)
        net = net.float()
        input_ele_tensor = input_ele_tensor.float()
        net.to(device)
        input_ele_tensor = input_ele_tensor.to(device)

        # loss_function = nn.MSELoss()
        # loss_function = MyMSE()
        # lamda = 10 ** level
        loss_function = MyLoss(args.lamda, args.k_s, args.mask)
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # epochs = ((lamda + 1) * 2 + 1) * 100
        epoch_list = list()
        loss_epoch = list()
        for epoch in range(args.epochs):
            running_loss = 0
            for step in range(args.steps):
                # 梯度置零
                optimizer.zero_grad()
                # 前后传播加优化
                output = net(input_ele_tensor)
                loss = loss_function(output, input_ele_tensor)
                loss.backward()
                optimizer.step()

                # 打印统计信息
                running_loss += loss.item()
                if step % args.steps == args.steps - 1:
                    print("lamda size:{0} epoch:{1} loss:{2}".format(args.lamda, epoch, running_loss / args.steps))
                if epoch == (args.epochs - 1) and step == (args.steps - 1):
                    torch.save(net.state_dict(), f'par/5_lamda{args.lamda}_{args.lr}_{args.con_c}.pt')
            epoch_list.append(epoch)
            loss_epoch.append(running_loss / args.steps)
        plt.plot(epoch_list, loss_epoch)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()
        torch.cuda.empty_cache()


def predict(input_ilr_tensor, clr_arr, multi, mask_arr):
    index = 2
    chosen_list_len = 39
    device = torch.device("cuda")
    # model = ConvAutoencoderOne(3, chosen_list_len - 1)
    # model = ConvAutoencoder(3, chosen_list_len - 1)
    model = ConvAutoencoderSample(args.k_s, args.in_len, base=args.con_c)
    # model = Autoencoder(chosen_list_len - 1)
    model.load_state_dict(torch.load(f'par/lamda{args.lamda}_{args.lr}.pt'))
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
        score = CalcAnomalScore(err, mode=args.mode)
        if args.mask:
            score_mask = score * mask_arr
            arr2raster(score_mask, 'mc/layer/mineral occurrence.shp',
                       f'img/result/score_{args.mode}_{args.lamda}_{args.mask}_{args.lr}.tif')
            recons_mask = recons_res[index] * mask_arr
            err = np.abs(np.subtract(recons_mask, clr_arr[index])) * mask_arr
        else:
            arr2raster(score, 'mc/layer/mineral occurrence.shp', f'img/phase/score_{args.mode}_{args.lamda}_{args.mask}.tif')
            recons = recons_res[index]
            err = np.abs(np.subtract(recons, clr_arr[index]))
    if not multi:
        err = np.abs(np.subtract(recons[0][0], clr_arr[index]))
        arr2raster(recons[0][0], 'mc/layer/mineral occurrence.shp', f'img/Au_recons_clr_one.tif')
        arr2raster(clr_arr[index], 'mc/layer/mineral occurrence.shp', f'img/Au_clr.tif')
        arr2raster(err, 'mc/layer/mineral occurrence.shp', f'img/Au_err_one.tif')
    else:
        if args.mask:
            arr2raster(recons_mask, 'mc/layer/mineral occurrence.shp', f'img/phase/mask/Au_recons_clr_{args.lamda}_{args.mask}.tif')
            arr2raster(clr_arr[index] * mask_arr, 'mc/layer/mineral occurrence.shp', f'img/phase/mask/Au_clr_mask.tif')
            arr2raster(err, 'mc/layer/mineral occurrence.shp', f'img/phase/mask/Au_err_{args.lamda}_{args.mask}_{args.lr}.tif')
        else:
            arr2raster(recons, 'mc/layer/mineral occurrence.shp',
                       f'img/phase/Au_recons_clr_{args.lamda}_{args.mask}.tif')
            arr2raster(clr_arr[index], 'mc/layer/mineral occurrence.shp', f'img/phase/Au_clr.tif')
            arr2raster(err, 'mc/layer/mineral occurrence.shp', f'img/phase/Au_err_{args.lamda}_{args.mask}.tif')


def predict_5(input_ilr_tensor, clr_arr, multi, mask_arr):
    index = 2
    chosen_list_len = 5
    device = torch.device("cuda")
    model = ConvAutoencoderSample(args.k_s, args.in_len, base=args.con_c)
    model.load_state_dict(torch.load(f'par/5_lamda{args.lamda}_{args.lr}_{args.con_c}.pt'))
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
        # 转clr
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
        err = np.subtract(recons[0], clr_arr[0])
        # err = err.reshape((chosen_list_len, 102, 82))
        score = CalcAnomalScore(err.cpu().detach().numpy(), mode=args.mode)
        if args.mask:
            score_mask = score * mask_arr
            arr2raster(score_mask, 'mc/layer/mineral occurrence.shp',
                       f'img/result/score_{args.mode}_{args.lamda}_{args.mask}_{args.lr}_{args.con_c}_5.tif')
            recons_mask = recons[0][index] * mask_arr
            # err = np.abs(np.subtract(recons_mask, clr_arr[index])) * mask_arr
        else:
            arr2raster(score, 'mc/layer/mineral occurrence.shp', f'img/phase/score_{args.mode}_{args.lamda}_{args.mask}.tif')
            recons = recons[0][index]
            err = np.abs(np.subtract(recons, clr_arr[index]))
    if not multi:
        err = np.abs(np.subtract(recons[0][0], clr_arr[index]))
        arr2raster(recons[0][0], 'mc/layer/mineral occurrence.shp', f'img/Au_recons_clr_one.tif')
        arr2raster(clr_arr[index], 'mc/layer/mineral occurrence.shp', f'img/Au_clr.tif')
        arr2raster(err, 'mc/layer/mineral occurrence.shp', f'img/Au_err_one.tif')
    else:
        if args.mask:
            arr2raster(recons_mask, 'mc/layer/mineral occurrence.shp', f'img/phase/mask/Au_recons_clr_{args.lamda}_'
                                                                       f'{args.mask}_{args.con_c}_5.tif')
            # arr2raster(clr_arr[index] * mask_arr, 'mc/layer/mineral occurrence.shp', f'img/phase/mask/Au_clr_mask.tif')
            # arr2raster(err, 'mc/layer/mineral occurrence.shp', f'img/phase/mask/Au_err_{args.lamda}_{args.mask}_{args.lr}.tif')
        else:
            arr2raster(recons, 'mc/layer/mineral occurrence.shp',
                       f'img/phase/Au_recons_clr_{args.lamda}_{args.mask}.tif')
            arr2raster(clr_arr[index], 'mc/layer/mineral occurrence.shp', f'img/phase/Au_clr.tif')
            arr2raster(err, 'mc/layer/mineral occurrence.shp', f'img/phase/Au_err_{args.lamda}_{args.mask}.tif')


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--in_len', type=int, default=5, help='input channels')
parser.add_argument('--k_s', type=int, default=3, help='kernel size')
parser.add_argument('--con_c', type=int, default=4, help='conv base')
parser.add_argument('--multi', type=bool, default=True, help='Multi channels or single channel')
parser.add_argument('--mask', type=bool, default=True, help='Mask or not')
parser.add_argument('--mode', type=str, default='all', help='score mode')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--steps', type=int, default=8, help='Number of optimize steps')
parser.add_argument('--lamda', type=float, default=1, help='Parameter lamda in loss function')
parser.add_argument('--seed', type=int, default=20, help='Seed number')
args = parser.parse_args()


if __name__ == '__main__':
    setup_seed(args.seed)
    input_ilr_tensor, clr_arr, mask_arr = data_Gen(args.mask)
    # input_ilr_tensor, clr_arr, mask_arr = data_Gen_5(args.mask)
    # train_one(index=2, input_ilr_tensor=input_ilr_tensor, clr_arr=clr_arr, multi=args.multi)
    # predict(input_ilr_tensor, clr_arr, multi=args.multi, mask_arr=mask_arr)

    # Au_nor = clr_arr.cpu().detach().numpy()
    # arr2raster(Au_nor[0][0], 'mc/layer/mineral occurrence.shp', f'img/phase/Ag_nor.tif')
    # arr2raster(Au_nor[0][1], 'mc/layer/mineral occurrence.shp', f'img/phase/As_nor.tif')
    # arr2raster(Au_nor[0][2], 'mc/layer/mineral occurrence.shp', f'img/phase/Au_nor.tif')
    # arr2raster(Au_nor[0][3], 'mc/layer/mineral occurrence.shp', f'img/phase/Bi_nor.tif')
    # arr2raster(Au_nor[0][4], 'mc/layer/mineral occurrence.shp', f'img/phase/Cu_nor.tif')

    # predict_5(input_ilr_tensor, clr_arr, multi=args.multi, mask_arr=mask_arr)

    # ROC(True)
