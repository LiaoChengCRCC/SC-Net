import shutil

import torch
# import tensorflow as tf
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import cv2
import random
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datetime import datetime
from data_augment import Mydataset
from model import RegNet_MultiHeadU3_2025,RegNet_MultiHeadU3_2026_noS,RegNet_MultiHeadU3_Land,RegNet_MultiHeadU3_Land_Con
from utils.seg_metric import Pixel_MA
from loss import dice_bce_loss_with_logits, dice_ctr_loss_with_logits,prcision_loss_with_logits
from loss0107 import LandObjContrastiveLoss

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# tf.logging.set_verbosity(tf.logging.INFO)
# pretrained_path = r'D:\road\Result\12-15_10-27-00\19_checkpoint-best.pth'
# pretrained_path = r'D:\road\Result\12-21_16-56-44_continue_7647\53_checkpoint-best.pth'
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class args:
    # train_path = r'F:\Datasets\LandSlide\UAV\res\train.csv'
    # val_path = r'F:\Datasets\LandSlide\UAV\res\test.csv'

    train_path = r'F:\Datasets\LandSlide\SAT\res\train.csv'
    val_path = r'F:\Datasets\LandSlide\SAT\res\test.csv'

    # train_path = r'F:\Datasets\LandSlide\GVLM\train.csv'
    # val_path = r'F:\Datasets\LandSlide\GVLM\test.csv'

    result_dir = 'Result/'
    batch_size = 16
    learning_rate = 0.01
    max_epoch = 100

if __name__ == '__main__':

    best_train_acc = 0.5
    seed_torch(1204)
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    # 模型保存路径
    log_dir = os.path.join(args.result_dir, time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)
    # normMean = [0.4758, 0.4873, 0.5098]
    # normStd = [0.1670, 0.1496, 0.1477]
    normMean = [0, 0, 0]
    normStd = [1, 1, 1]
    normTransfrom = transforms.Normalize(normMean, normStd)
    transform_edge = transforms.Compose([
        transforms.ToTensor()
    ])

    transform_rgb = transforms.Compose([
        transforms.ToTensor(),
        normTransfrom
    ])
    # # 数据加载，详见data_agu.py best:Mydataset
    train_data = Mydataset(path=args.train_path, transformrgb=transform_rgb, transformedge=transform_edge, augment=True)
    val_data = Mydataset(path=args.val_path, transformrgb=transform_rgb, transformedge=transform_edge, augment=False)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    print("Num of training data:", len(train_loader) * args.batch_size)
    print("Num of validation data:", len(val_loader) * args.batch_size)

    # net = RegNet_MultiHeadU3_Land().cuda()
    net = RegNet_MultiHeadU3_Land_Con().cuda()

    # 损失函数及优化方法定义
    criterion4 = dice_bce_loss_with_logits().cuda()
    contrastloss = LandObjContrastiveLoss().cuda()
    # criterion_ctr = dice_ctr_loss_with_logits().cuda()
    # criterion_pre = prcision_loss_with_logits().cuda()
    # criterion = MultiLevelCrackLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, dampening=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, verbose=True, min_lr=0.0000001)
    savepath = log_dir
    # ---------------------------4、训练网络---------------------------
    for epoch in range(args.max_epoch):

        loss_sigma = 0.0
        loss_val_sigma = 0.0
        acc_val_sigma = 0.0
        net.train()

        for i, data in enumerate(train_loader):
            inputs, inputs2, labels,  lab_name = data
            inputs = Variable(inputs.cuda())
            inputs2 = Variable(inputs2.float().cuda())
            labels = Variable(labels.cuda())
            labels = labels.float().cuda()

            optimizer.zero_grad()
            outputs, feat = net.forward(inputs, inputs2)
            outputs = torch.squeeze(outputs, dim=1)

            # loss = criterion4(labels, outputs)
            loss1 = criterion4(labels, outputs)
            loss2 = contrastloss(feat, labels)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            loss_sigma += loss.item()
            if i % 50 == 0 and i > 0:
                loss_avg = loss_sigma / 50
                loss_sigma = 0.0
                print("Training:Epoch[{:0>3}/{:0>3}] Iter[{:0>3}/{:0>3}] Loss:{:.4f}".format(
                    epoch + 1, args.max_epoch, i + 1, len(train_loader), loss_avg))
                writer.add_scalar("LOSS", loss_avg, epoch)

        if not os.path.exists(savepath):
            os.mkdir(savepath)
        tmp_save_name = os.path.join(savepath, str(epoch))
        print(tmp_save_name)
        if not os.path.exists(tmp_save_name):
            os.mkdir(tmp_save_name)
        # ---------------------------每个epoch验证网络---------------------------
        TP = 0
        FP = 0
        FN = 0
        TP0 = 0
        FP0 = 0
        FN0 = 0
        if epoch % 1 == 0:
            net.eval()
            acc_val_sigma = 0
            acc_val = 0
            data_list = []
            for i, data in enumerate(val_loader):
                inputs, inputs2, labels,  img_name = data
                inputs = Variable(inputs.cuda())
                inputs2 = Variable(inputs2.float().cuda())
                labels = Variable(labels.cuda())
                labels = labels.float().cuda()
                with torch.no_grad():
                    predicts,_ = net.forward(inputs, inputs2)
                    # predicts = net.forward(inputs)

                # predicts = torch.cat(predicts, dim=0)
                predicts = torch.sigmoid(predicts)
                predicts[predicts < 0.5] = 0
                predicts[predicts >= 0.5] = 1
                result = np.squeeze(predicts)
                # outputs = torch.squeeze(outputs, dim=1)

                cc = labels.shape[0]
                for index in range(cc):
                    # 评估方法为平均iou
                    cv2.imwrite(os.path.join(tmp_save_name, img_name[index]),
                                result[index].cpu().detach().numpy() * 255)
                    # tp, fp, fn = Pixel_A(result[index].cpu().detach().numpy(), labels[index].cpu().detach().numpy())
                    tp, fp, fn, tp0, fp0, fn0, = Pixel_MA(result[index].cpu().detach().numpy(),
                                                          labels[index].cpu().detach().numpy())

                    TP += tp
                    FP += fp
                    FN += fn
                    TP0 += tp0
                    FP0 += fp0
                    FN0 += fn0
                    # acc_val_sigma += mean_IU(labels[index].cpu().detach().numpy(), result[index].cpu().detach().numpy())

            # 验证精度提高时，保存模型
            # val_acc = acc_val_sigma / args.num_test_img
            F1 = 2 * TP / (2 * TP + FP + FN + 1e-6)
            F1_0 = 2 * TP0 / (2 * TP0 + FP0 + FN0 + 1e-6)
            anval_p = TP / (TP + FP + 1e-6)
            anval_r = TP / (TP + FN + 1e-6)
            val_iou = anval_p * anval_r / (anval_p + anval_r - anval_p * anval_r)

            anval_p0 = TP0 / (TP0 + FP0)
            anval_r0 = TP0 / (TP0 + FN0)
            val_iou0 = anval_p0 * anval_r0 / (anval_p0 + anval_r0 - anval_p0 * anval_r0)

            print("valid F1:", F1, ", F1_0:", F1_0)
            print("Mean F1_0:", (F1 + F1_0) / 2.0)

            print("valid_IoU1:", val_iou, ", IoU:", val_iou0)
            print("Mean IoU:", (val_iou + val_iou0) / 2.0)
            print("Precision:", anval_p, "Recall:", anval_r)
            print("----------------------------------")
            print("best F1:", best_train_acc)
            scheduler.step(F1)
            if (F1) > best_train_acc:
                # state = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
                state = {'state_dict': net.state_dict()}
                filename = os.path.join(log_dir, str(epoch) + '_checkpoint-best.pth')
                torch.save(state, filename)
                best_train_acc = F1
            else:
                shutil.rmtree(tmp_save_name)

    writer.close()
    net_save_path = os.path.join(log_dir, 'net_params_end.pkl')
    torch.save(net.state_dict(), net_save_path)