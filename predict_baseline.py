import os
import time

import cv2
import math
import glob
import torch
import json
import shutil
import numpy as np
import pandas as pd
# from skimage import morphology
import torch.nn.functional as F
from torchvision import transforms as T
from utils.seg_metric import Pixel_MA
from skimage import morphology
from torch.nn.parallel import DataParallel
from utils.data_aug_crack import rgb_2_grad3
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from data_agu import Mydataset
# from DLinkNet import Net_TesUU
from model_sat import RegNet_MultiHeadU3_Land, RegNet_MultiHeadU3_Land_Con

# pretrained_path = r'D:\road\Result\11-18_16-41-50_9k_741\44_checkpoint-best.pth'
# pretrained_path = r'D:\road\Result\12-16_17-36-49\49_checkpoint-best.pth'
# pretrained_path = r'D:\road\Result\12-17_21-15-12\51_checkpoint-best.pth'
# pretrained_path = r'D:\road\Result\12-21_16-56-44_continue_7647\53_checkpoint-best.pth'
# pretrained_path = r'D:\road\Result\12-31_15-33-49\models\49_checkpoint-best.pth'

pretrained_path = r'C:\Users\Administrator\PycharmProjects\CrackSeg\Result\CAS-UVA\01-10_09-43-39\90_checkpoint-best.pth'
# pretrained_path = r'C:\Users\Administrator\PycharmProjects\CrackSeg\Result\CAS-SAT\01-16_07-59-41\79_checkpoint-best.pth'


# normMean = [0.4758, 0.4873, 0.5098]
# normStd = [0.1670, 0.1496, 0.1477]
# normTransfrom = transforms.Normalize(normMean, normStd)
# transform = transforms.Compose([
#         transforms.ToTensor(),
#         normTransfrom
#     ])


def truncated_linear_stretch(image, truncated_value=0.5, max_out=255, min_out=0, back_ignore=True):
    def gray_process(gray, dth, uth, maxout=max_out, minout=min_out):
        truncated_down = np.percentile(gray, truncated_value + dth)
        truncated_up = np.percentile(gray, 100 - truncated_value - uth)
        gray_new = (gray - truncated_down) / (truncated_up - truncated_down) * (maxout - minout) + minout
        gray_new[gray_new < minout] = minout
        gray_new[gray_new > maxout] = maxout
        return np.uint8(gray_new)

    # ignore the background values of 0 and 255
    back_dth = 0
    back_uth = 0
    if back_ignore:
        back_dth = (100 - truncated_value) * np.sum(image[:, :, 0] == min_out) / (image.shape[0] * image.shape[1])
        back_uth = (100 - truncated_value) * np.sum(image[:, :, 0] == max_out) / (image.shape[0] * image.shape[1])

    if image.shape[2] == 4:
        (b, g, r, l) = cv2.split(image)
        b = gray_process(b, back_dth, back_uth)
        g = gray_process(g, back_dth, back_uth)
        r = gray_process(r, back_dth, back_uth)
        l = gray_process(l, back_dth, back_uth)
        result = cv2.merge((b, g, r, l))
        return result
    else:
        (b, g, r) = cv2.split(image)
        b = gray_process(b, back_dth, back_uth)
        g = gray_process(g, back_dth, back_uth)
        r = gray_process(r, back_dth, back_uth)
        result = cv2.merge((b, g, r))
        return result


def normalize_to_255(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized = (image - min_val) / (max_val - min_val)
    scaled = (normalized * 255).astype(np.uint8)
    return scaled

class Mydataset(Dataset):
    def __init__(self, path,augment=False,transformrgb=None, transformedge=None, target_transform=None):

        self.aug=augment
        self.file_path=os.path.dirname(path)
        self.img_size=512 #448
        # self.img_size = 448
        data = pd.read_csv(path)  # 获取csv表中的数据
        imgs = []
        for i in range(len(data)):
            imgs.append((data.iloc[i,0], data.iloc[i,1]))
        np.random.shuffle(imgs)
        self.imgs = imgs
        self.transform_rgb = transformrgb
        self.transform_dge = transformedge
        self.target_transform = target_transform

    def __getitem__(self, item):
        if self.aug==False:
            fn, lab = self.imgs[item]
            # fn = os.path.join(self.file_path, "image_A/" + fn)
            # label = os.path.join(self.file_path, "image_A/" + lab)
            fn = os.path.join(self.file_path, "images/"+ fn)
            label = os.path.join(self.file_path, "masks/"+ lab)

            bgr_img = cv2.imread(fn, -1)
            rgb_img = bgr_img[..., ::-1]  # bgr2rgb
            rgb_img = cv2.resize(rgb_img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            # bgr_img = truncated_linear_stretch(rgb_img, truncated_value=0.5, max_out=255, min_out=0, back_ignore=True)
            gt = cv2.imread(label, -1) // 255
            gt = cv2.resize(gt, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

            gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            # grad = (255 * grade(gray)).astype(np.uint8)
            # grad_img = rgb2grad(gray)
            grad_img = rgb_2_grad3(gray)



            # img = Image.open(fn).convert('RGB')
            # img = cv2.merge([rgb_img, grad])
            img = Image.fromarray(rgb_img)
            if self.transform_rgb is not None:
                img = self.transform_rgb(img)

            # grad_img = grad_img.transpose(1,2,0).astype(np.float32)
            if self.transform_dge is not None:
                grad_img = self.transform_dge(grad_img)

            # grad_img=grad_img.transpose(2, 0, 1).astype(np.float32)
            # grad_img = torch.tensor(grad_img/255.0).float()

            # ctr_line = skeletonize(gt > 0).astype(np.uint8)

            # k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # ctr_line = cv2.dilate(ctr_line, k,iterations=1)
            return img,grad_img, gt, lab

        else:
            # 进行数据增强
            fn, lab = self.imgs[item]
            # train with data.cvs
            fn = os.path.join(self.file_path, "images/"+ fn)
            label = os.path.join(self.file_path, "masks/"+ lab)

            gt = cv2.imread(label, -1)//255
            # gt = cv2.imread(label, -1)
            image = cv2.imread(fn,-1)

            rgb_img = image[..., ::-1]  # bgr2rgb
            # image = truncated_linear_stretch(rgb_img, truncated_value=0.5, max_out=255, min_out=0, back_ignore=True)

            gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            # grad = (255 * grade(gray)).astype(np.uint8)
            grad_img = rgb_2_grad3(gray)
            # grad_img = rgb2grad(gray)

            # img = cv2.merge([rgb_img, grad])
            img = Image.fromarray(rgb_img)
            if self.transform_rgb is not None:
                img = self.transform_rgb(img.copy())

            # grad_img=grad_img.transpose(2, 0, 1).astype(np.float32)
            # grad_img = torch.tensor(grad_img / 255.0).float()

            # grad_img = grad_img.transpose(1,2,0).astype(np.float32)
            # grad_img = Image.fromarray(grad_img)
            if self.transform_dge is not None:
                grad_img = self.transform_dge(grad_img)

            return img, grad_img, gt.copy(), lab

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    IS_HALF = False
    PADDING = 0
    # batch_size = 2
    batch_size = 4
    # scales = [1.0, 1.25, 1.5]
    scales = [1.5]

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

    st = time.time()

    # ori_image_path=r'D:\CrackData\images'
    # dst_image_path=r'D:\CrackData\clip_samples\images'
    # ori_label_path = r'D:\CrackData\masks'
    # dst_label_path = r'D:\CrackData\clip_samples\labels'
    # val_path = r'D:\CrackData\lfeng\test.csv'

    # val_path = r'F:\Datasets\LandSlide\UAV\res\test.csv'
    # val_path = r'F:\Datasets\LandSlide\SAT\res\test.csv'
    # val_path = r'F:\Datasets\LandSlide\SAT\res\test1.csv'
    val_path = r'F:\Datasets\LandSlide\GVLM\test.csv'
    tmp_save_name = r'C:\Users\Administrator\PycharmProjects\CrackSeg\Result\TRANS\UAV-G'

    val_data = Mydataset(path=val_path, transformrgb=transform_rgb, transformedge=transform_edge, augment=False)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    # val_data = Mydataset(path=val_path, transform=transform, augment=False)
    # val_loader = DataLoader(dataset=val_data, batch_size=4, shuffle=False, drop_last=True)

    net_se = RegNet_MultiHeadU3_Land_Con().cuda()  # valid acc: 0.5787757893149679
    # net_se = DataParallel(net_se)

    # net_se = RegNet_MultiHeadU3()
    # net_se.cuda()

    if torch.cuda.is_available():
        # continue training...
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(pretrained_path)
        net_se = net_se.to(device)
        net_se.load_state_dict(checkpoint['state_dict'])
    net_se.eval()

    if IS_HALF:
        net_se.half()
    net_se.to(device)

    TP = 0
    FP = 0
    FN = 0
    TP0 = 0
    FP0 = 0
    FN0 = 0

    for i, data in enumerate(val_loader):
        inputs, inputs2, labels, img_name = data
        inputs = Variable(inputs.cuda())
        inputs2 = Variable(inputs2.cuda())
        labels = Variable(labels.cuda())
        labels = labels.float().cuda()
        with torch.no_grad():
            predicts,_ = net_se.forward(inputs, inputs2)
        # valid F1: 0.7890896726393345, valid_IoU1: 0.6516499651623723/448
        # valid F1: 0.7905547250865971, valid_IoU1: 0.6536506789389056/512

        predicts = torch.sigmoid(predicts)
        predicts[predicts < 0.5] = 0  # 0.65
        predicts[predicts >= 0.5] = 1
        result = np.squeeze(predicts)
        # outputs = torch.squeeze(outputs, dim=1)

        # h = h.numpy()
        # w = w.numpy()
        cc = labels.shape[0]
        for index in range(cc):
            res = result[index].cpu().detach().numpy()
            # res = cv2.resize(res, (w[index], h[index]), interpolation=cv2.INTER_NEAREST)

            cv2.imwrite(os.path.join(tmp_save_name, img_name[index]), res * 255)


            tp, fp, fn, tp0, fp0, fn0, = Pixel_MA(res, labels[index].cpu().detach().numpy())

            ### tp, fp, fn = Pixel_A(result[index].cpu().detach().numpy(), labels[index].cpu().detach().numpy())
            # tp, fp, fn, tp0, fp0, fn0, = Pixel_MA(result[index].cpu().detach().numpy(),
            #                                       labels[index].cpu().detach().numpy())

            # F1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
            # if F1 < 0.3 and np.sum(labels[index].cpu().detach().numpy())>32: #np.sum(labels[index].cpu().detach().numpy())>32 and
            #     print(img_name[index])
            #     shutil.move(os.path.join(ori_image_path, img_name[index]), os.path.join(dst_image_path, img_name[index]))
            #     shutil.move(os.path.join(ori_label_path, img_name[index]), os.path.join(dst_label_path, img_name[index]))

            TP += tp
            FP += fp
            FN += fn
            TP0 += tp0
            FP0 += fp0
            FN0 += fn0
            #### acc_val_sigma += mean_IU(labels[index].cpu().detach().numpy(), result[index].cpu().detach().numpy())

    F1 = 2 * TP / (2 * TP + FP + FN)
    F1_0 = 2 * TP0 / (2 * TP0 + FP0 + FN0)
    anval_p = TP / (TP + FP)
    anval_r = TP / (TP + FN)
    val_iou = anval_p * anval_r / (anval_p + anval_r - anval_p * anval_r)

    anval_p0 = TP0 / (TP0 + FP0)
    anval_r0 = TP0 / (TP0 + FN0)
    val_iou0 = anval_p0 * anval_r0 / (anval_p0 + anval_r0 - anval_p0 * anval_r0)

    print("anval_p:", anval_p, ", anval_r:", anval_r)

    print("valid F1:", F1, ", F1_0:", F1_0)
    print("Mean F1_0:", (F1 + F1_0) / 2.0)

    print("valid_IoU1:", val_iou, ", IoU:", val_iou0)
    print("Mean IoU:", (val_iou + val_iou0) / 2.0)

    print('total time cost:', time.time() - st)