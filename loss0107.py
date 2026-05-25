"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py
"""

from torch.autograd import Variable
import torch.nn.functional as F
import torch
import cv2
import torch.nn as nn
import numpy as np
import pycocotools.mask as mutils
from scipy.ndimage import distance_transform_edt as distance

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res


def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)


def surface_loss_keras(y_true, y_pred):
    # y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
    #                                  inp=[y_true],
    #                                  Tout=tf.float32)
    y_true_dist_map = calc_dist_map(y_true)
    multipled = y_pred * y_true_dist_map
    return multipled


class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        # score = (2. * intersection + smooth) / (i + j + smooth)
        score = (intersection + smooth) / (i + j - intersection + smooth)  # iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a + b


# class dice_bce_loss_with_logits(nn.Module):
#     def __init__(self, batch=True):
#         super(dice_bce_loss_with_logits, self).__init__()
#         self.batch = batch
#         # self.bce_loss = nn.BCELoss()
#         # self.bce_loss = F.binary_cross_entropy_with_logits()
#
#     def soft_dice_coeff(self, y_true, y_pred):
#         y_pred = torch.sigmoid(y_pred)
#         smooth = 0.0  # may change
#         # smooth = 1.0  # may change
#         # y_true = y_true1 + y_true2
#         if self.batch:
#
#             i = torch.sum(y_true)
#             j = torch.sum(y_pred)
#             intersection = torch.sum(y_true * y_pred)
#         else:
#             i = y_true.sum(1).sum(1).sum(1)
#             j = y_pred.sum(1).sum(1).sum(1)
#             intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
#         score = (2. * intersection + smooth) / (i + j + smooth)
#         # score = (intersection + smooth) / (i + j - intersection + smooth) #iou
#         return score.mean()
#
#     def soft_dice_loss(self, y_true, y_pred):
#         loss = 1 - self.soft_dice_coeff(y_true, y_pred)
#         return loss
#
#     # def __call__(self, y_true1, y_pred1,y_true2, y_pred2,y_pred):
#     def __call__(self, y_true, y_pred):
#
#         a = F.binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=torch.Tensor([2.5]).cuda())
#         b = self.soft_dice_loss(y_true, y_pred)
#         return a + b
#
#
# class dice_bce_loss_with_logits2(nn.Module):
#     def __init__(self, batch=True):
#         super(dice_bce_loss_with_logits2, self).__init__()
#         self.batch = batch
#         # self.bce_loss = nn.BCELoss()
#         # self.bce_loss = F.binary_cross_entropy_with_logits()
#
#     def soft_dice_coeff(self, y_true, y_pred):
#         y_pred = torch.sigmoid(y_pred)
#
#         loss_w = (y_true < 2).float()
#         y_pred = loss_w * y_pred
#         y_true = loss_w * y_true
#
#         smooth = 0.0  # may change
#         # smooth = 1.0  # may change
#         # y_true = y_true1 + y_true2
#         if self.batch:
#
#             i = torch.sum(y_true)
#             j = torch.sum(y_pred)
#             intersection = torch.sum(y_true * y_pred)
#         else:
#             i = y_true.sum(1).sum(1).sum(1)
#             j = y_pred.sum(1).sum(1).sum(1)
#             intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
#         score = (2. * intersection + smooth) / (i + j + smooth)
#         # score = (intersection + smooth) / (i + j - intersection + smooth) #iou
#         return score.mean()
#
#     def soft_dice_loss(self, y_true, y_pred):
#         loss = 1 - self.soft_dice_coeff(y_true, y_pred)
#         return loss
#
#     # def __call__(self, y_true1, y_pred1,y_true2, y_pred2,y_pred):
#     def __call__(self, y_true, y_pred):
#         # loss_w = y_true
#         loss_w = (y_true < 2).float()
#         # f_lab = 1 - y_true
#         # dis = torch.pairwise_distance(feat1, feat2)
#         # c = torch.sum(torch.multiply(f_lab, dis)) / torch.sum(f_lab)
#         # y_true = y_true1 + y_true2
#         # a = F.binary_cross_entropy_with_logits(y_pred, y_true)
#         a = F.binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=torch.Tensor([2.5]).cuda(), weight=loss_w)
#
#         # one_true=255*np.ones_like(y_true.cpu())
#         # one_true[y_true1.cpu() == 1] = 1
#         # one_true[y_true2.cpu() == 1] = 0
#         # y1_true=one_true
#         # b = nn.CrossEntropyLoss(y_pred1, y1_true,ignore_index=255)
#         # a = self.bce_loss(y_pred, y_true)
#
#         # return a
#         b = self.soft_dice_loss(y_true, y_pred)
#         return a + b

class dice_bce_loss_with_logits(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss_with_logits, self).__init__()
        self.batch = batch
        # self.bce_loss = nn.BCELoss()
        # self.bce_loss = F.binary_cross_entropy_with_logits()

    def soft_dice_coeff(self, y_true, y_pred):
        y_pred = torch.sigmoid(y_pred)
        # smooth = 0.0  # may change
        smooth = 1.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth) #iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        a = F.binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=torch.Tensor([1.5]).cuda())
        # a = nn.BCEWithLogitsLoss(y_pred, y_true, pos_weight=torch.Tensor([1.5]).cuda())
        # a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return 2*a+b

class trky_bce_loss_with_logits(nn.Module):
    def __init__(self, batch=True):
        super(trky_bce_loss_with_logits, self).__init__()
        self.batch = batch
        # self.bce_loss = nn.BCELoss()
        # self.bce_loss = F.binary_cross_entropy_with_logits()

        # y_true_pos = K.flatten(y_true)
        # y_pred_pos = K.flatten(y_pred)
        # true_pos = K.sum(y_true_pos * y_pred_pos)
        # false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
        # false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
        # alpha = 0.7
        # return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

    def soft_dice_coeff(self, y_true, y_pred):
        y_pred = torch.sigmoid(y_pred)
        smooth = 0.0  # may change
        # smooth = 1.0  # may change
        # y_true = y_true1 + y_true2
        if self.batch:

            i = torch.sum(y_true * (1 - y_pred))
            j = torch.sum(y_pred * (1 - y_true))
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        alpha = 0.7
        score = (intersection + smooth) / (alpha * i + (1 - alpha) * j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth) #iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    # def __call__(self, y_true1, y_pred1,y_true2, y_pred2,y_pred):
    def __call__(self, y_true, y_pred):

        # f_lab = 1 - y_true
        # dis = torch.pairwise_distance(feat1, feat2)
        # c = torch.sum(torch.multiply(f_lab, dis)) / torch.sum(f_lab)
        # y_true = y_true1 + y_true2
        # a = F.binary_cross_entropy_with_logits(y_pred, y_true)
        y_true = y_true.float()
        # a = F.binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=torch.Tensor([2.5]).cuda())
        # one_true=255*np.ones_like(y_true.cpu())
        # one_true[y_true1.cpu() == 1] = 1
        # one_true[y_true2.cpu() == 1] = 0
        # y1_true=one_true
        # b = nn.CrossEntropyLoss(y_pred1, y1_true,ignore_index=255)
        # a = self.bce_loss(y_pred, y_true)

        # return a
        b = self.soft_dice_loss(y_true, y_pred)
        return b


def Raster2poly(rasterlab):
    bbox = []
    lab = rasterlab.astype(np.uint8)
    nc, label = cv2.connectedComponents(lab, connectivity=8)
    for c in range(nc):
        if np.all(lab[label == c] == 0):
            continue
        else:
            ann = np.asfortranarray((label == c).astype(np.uint8))
            rle = mutils.encode(ann)
            if (mutils.area(rle)) < 64: #128 for SAT, 256 for UVA, 128 for GVLM
                continue
            seg_contours, _ = cv2.findContours(ann, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for seg in seg_contours:
                seginit_coord = np.array(seg.transpose(1, 0, 2))
                # if seginit_coord
                bbox.append(seginit_coord)
    return bbox


# class LandObjContrastiveLoss(nn.Module):
#     def __init__(self):
#         super(LandObjContrastiveLoss, self).__init__()
#         self.bbox = []
#
#     def __call__(self, feat_all, lab_b):
#         losses_seg = 0.0
#         sum_seg = 0
#         all_bboxs = []
#         lab_b = torch.squeeze(F.interpolate(torch.unsqueeze(lab_b, dim=0), scale_factor=0.25, mode='nearest'))
#         lab_b[lab_b > 1] = 0
#         for index in range(16):  # convert to polygon :batch_size
#             lb = lab_b[index, :, :].cpu().detach().numpy()
#             all_bboxs.append(Raster2poly(lb))
#
#         for index in range(15):
#             if len(all_bboxs[index]) > 0:
#                 for j in range(len(all_bboxs[index])):
#                     bi = np.zeros((128, 128), dtype=np.uint8)
#                     bi = cv2.fillPoly(bi, all_bboxs[index][j], 1)
#                     for k in range(15-index):
#                         if len(all_bboxs[index + 1 + k]) > 0:
#                             bj = np.zeros((128, 128), dtype=np.uint8)
#                             bj = cv2.fillPoly(bj, all_bboxs[index + 1 + k][0], 1)
#                             feati1 = torch.multiply(feat_all[index, :, :, :],
#                                                     torch.unsqueeze(torch.from_numpy(bi).cuda(), dim=0))
#                             feati2 = torch.multiply(feat_all[index + 1 + k, :, :, :],
#                                                     torch.unsqueeze(torch.from_numpy(bj).cuda(), dim=0))
#                             # losses_seg += (torch.mean(0.5 + 0.5 * torch.cosine_similarity(feati1, feati2, dim=1)))
#                             losses_seg += (torch.mean(0.5 - 0.5 * torch.cosine_similarity(feati1, feati2, dim=1)))
#                             sum_seg += 1
#                         else: # for the background, cult max
#                             feati1 = torch.multiply(feat_all[index, :, :, :],
#                                                     torch.unsqueeze(torch.from_numpy(bi).cuda(), dim=0))
#                             feati2 = torch.multiply(feat_all[index + 1 + k, :, :, :],
#                                                     torch.unsqueeze(torch.from_numpy(bi).cuda(), dim=0))
#                             losses_seg += (torch.mean(0.5 + 0.5 * torch.cosine_similarity(feati1, feati2, dim=1)))
#                             sum_seg += 1
#         print('contrast_pairs:',sum_seg)
#         return losses_seg / sum_seg if sum_seg != 0 else 0.0

class LandObjContrastiveLoss(nn.Module):
    def __init__(self):
        super(LandObjContrastiveLoss, self).__init__()
        self.bbox = []

    def __call__(self, feat_all, lab_b):
        losses_seg = 0.0
        sum_seg = 0
        all_bboxs = []
        lab_b = torch.squeeze(F.interpolate(torch.unsqueeze(lab_b, dim=0), scale_factor=0.25, mode='nearest'))
        lab_b[lab_b > 1] = 0
        for index in range(16):  # convert to polygon :batch_size
            lb = lab_b[index, :, :].cpu().detach().numpy()
            all_bboxs.append(Raster2poly(lb))

        for index in range(15):
            if len(all_bboxs[index]) > 0:
                for j in range(len(all_bboxs[index])):
                    bi = np.zeros((128, 128), dtype=np.uint8)
                    bi = cv2.fillPoly(bi, all_bboxs[index][j], 1)
                    for k in range(15-index):
                        if len(all_bboxs[index + 1 + k]) > 0:
                            bj = np.zeros((128, 128), dtype=np.uint8)
                            bj = cv2.fillPoly(bj, all_bboxs[index + 1 + k][0], 1)
                            feat_flat1 = feat_all[index, :, :, :].view(-1, 144)
                            feati1 = feat_flat1[torch.from_numpy(bi).cuda().view(-1)]
                            vec1 = feati1.mean(dim=0)

                            feat_flat2 = feat_all[index + 1 + k, :, :, :].view(-1, 144)
                            feati2 = feat_flat2[torch.from_numpy(bj).cuda().view(-1)]
                            vec2 = feati2.mean(dim=0)
                            # losses_seg += (torch.mean(0.5 - 0.5 * torch.cosine_similarity(vec1, vec2, dim=0)))
                            losses_seg += 0.5 - 0.5 * torch.cosine_similarity(vec1, vec2, dim=0)
                            sum_seg += 1
                        # else: # for the background, cult max
                        #
                        #     feat_flat1 = feat_all[index, :, :, :].view(-1, 144)
                        #     feati1 = feat_flat1[torch.from_numpy(bi).cuda().view(-1)]
                        #     vec1 = feati1.mean(dim=0)
                        #
                        #     feat_flat2 = feat_all[index + 1 + k, :, :, :].view(-1, 144)
                        #     feati2 = feat_flat2[torch.from_numpy(bi).cuda().view(-1)]
                        #     vec2 = feati2.mean(dim=0)
                        #     losses_seg += 0.5 + 0.5 * torch.cosine_similarity(vec1, vec2, dim=0)
                        #     sum_seg += 1
        # print('contrast_pairs:',sum_seg)
        return losses_seg / sum_seg if sum_seg != 0 else 0.0

        # mask1 = torch.randint(0, 2, (W, H)).bool()
        # mask2 = torch.randint(0, 2, (W, H)).bool()
        # feat_flat = feat.view(-1, C)
        #
        # select_feat1 = feat_flat[mask1.view(-1)]
        # select_feat2 = feat_flat[mask2.view(-1)]
        #
        # vec1 = select_feat1.mean(dim=0)
        # vec2 = select_feat2.mean(dim=0)
        # dis = torch.cosine_similarity(vec1, vec2, dim=0)
        #

        #     for i in range(len(all_bboxs[index])):
        #         bi = np.zeros((128, 128), dtype=np.uint8)
        #         bi = cv2.fillPoly(bi, all_bboxs[index][i], 1)
        #         feati1 = torch.multiply(feat_all[index, :, :, :], torch.unsqueeze(torch.from_numpy(bi).cuda(), dim=0))
        #         feati2 = torch.multiply(feat_mov[index, :, :, :], torch.unsqueeze(torch.from_numpy(bi).cuda(), dim=0))
        #         # losses_seg += (1-torch.sum(torch.pairwise_distance(feati1, feati2,keepdim=True))/(64*torch.sum(torch.from_numpy(bi).cuda())))
        #         losses_seg += (torch.mean(0.5 + 0.5*torch.cosine_similarity(feati1, feati2,dim=1)))
        #         # losses_seg += torch.sum(1-torch.cosine_similarity(feati1, feati2, dim=0))/torch.sum(torch.from_numpy(bi).cuda())
        #         sum_seg += 1
        # return losses_seg / sum_seg if sum_seg != 0 else 0.0


class ObjContrastiveLoss2(nn.Module):
    def __init__(self):
        super(ObjContrastiveLoss2, self).__init__()
        self.bbox = []

    # def __call__(self, feat_all, feat_mov, lab_b):
    #     losses_seg = 0.0
    #     sum_seg = 0
    #     all_bboxs = []
    #     # mov_bboxs = []
    #     lab_b = torch.squeeze(F.interpolate(torch.unsqueeze(lab_b, dim=0), scale_factor=0.25, mode='nearest'))
    #     for index in range(8):
    #         lb = lab_b[index, :, :].cpu().detach().numpy()
    #         all_bboxs.append(Raster2poly(lb))
    #         # lm = lab_m[index, :, :].cpu().detach().numpy()
    #         # mov_bboxs.append(Raster2poly(lm))
    #
    #     for index in range(8):
    #         feati1 = feat_all[index, :, :, :]
    #         feati2 = feat_mov[index, :, :, :]
    #         losses_seg += (torch.mean(0.5 - 0.5 * torch.cosine_similarity(feati1, feati2, dim=1)))
    #         sum_seg += 1
    #
    #     return losses_seg / sum_seg if sum_seg != 0 else 0.0

    def __call__(self, feat_all, feat_mov, lab_b):
        losses_seg = 0.0
        sum_seg = 0
        all_bboxs = []
        # mov_bboxs = []
        lab_b = torch.squeeze(F.interpolate(torch.unsqueeze(lab_b, dim=0), scale_factor=0.25, mode='nearest'))
        lab_b[lab_b > 1] = 0
        for index in range(4):  # batch_size
            lb = lab_b[index, :, :].cpu().detach().numpy()
            all_bboxs.append(Raster2poly(lb))
            # lm = lab_m[index, :, :].cpu().detach().numpy()
            # mov_bboxs.append(Raster2poly(lm))

        for index in range(4):
            for i in range(len(all_bboxs[index])):
                bi = np.zeros((128, 128), dtype=np.uint8)
                bi = cv2.fillPoly(bi, all_bboxs[index][i], 1)
                feati1 = torch.multiply(feat_all[index, :, :, :], torch.unsqueeze(torch.from_numpy(bi).cuda(), dim=0))
                feati2 = torch.multiply(feat_mov[index, :, :, :], torch.unsqueeze(torch.from_numpy(bi).cuda(), dim=0))
                # losses_seg += (1-torch.sum(torch.pairwise_distance(feati1, feati2,keepdim=True))/(64*torch.sum(torch.from_numpy(bi).cuda())))
                losses_seg += (torch.mean(0.5 + 0.5 * torch.cosine_similarity(feati1, feati2, dim=1)))
                # losses_seg += torch.sum(1-torch.cosine_similarity(feati1, feati2, dim=0))/torch.sum(torch.from_numpy(bi).cuda())
                sum_seg += 1
        return losses_seg / sum_seg if sum_seg != 0 else 0.0


class ObjContrastiveLoss(nn.Module):
    def __init__(self):
        super(ObjContrastiveLoss, self).__init__()
        self.bbox = []

    # def __call__(self, feat_all, feat_mov, lab_b):
    #     losses_seg = 0.0
    #     sum_seg = 0
    #     all_bboxs = []
    #     # mov_bboxs = []
    #     lab_b = torch.squeeze(F.interpolate(torch.unsqueeze(lab_b, dim=0), scale_factor=0.25, mode='nearest'))
    #     for index in range(8):
    #         lb = lab_b[index, :, :].cpu().detach().numpy()
    #         all_bboxs.append(Raster2poly(lb))
    #         # lm = lab_m[index, :, :].cpu().detach().numpy()
    #         # mov_bboxs.append(Raster2poly(lm))
    #
    #     for index in range(8):
    #         feati1 = feat_all[index, :, :, :]
    #         feati2 = feat_mov[index, :, :, :]
    #         losses_seg += (torch.mean(0.5 - 0.5 * torch.cosine_similarity(feati1, feati2, dim=1)))
    #         sum_seg += 1
    #
    #     return losses_seg / sum_seg if sum_seg != 0 else 0.0

    def __call__(self, feat_all, feat_mov, lab_b):
        losses_seg = 0.0
        sum_seg = 0
        all_bboxs = []
        # mov_bboxs = []
        lab_b = torch.squeeze(F.interpolate(torch.unsqueeze(lab_b, dim=0), scale_factor=0.25, mode='nearest'))
        for index in range(4):
            lb = lab_b[index, :, :].cpu().detach().numpy()
            all_bboxs.append(Raster2poly(lb))
            # lm = lab_m[index, :, :].cpu().detach().numpy()
            # mov_bboxs.append(Raster2poly(lm))

        for index in range(4):
            for i in range(len(all_bboxs[index])):
                bi = np.zeros((128, 128), dtype=np.uint8)
                bi = cv2.fillPoly(bi, all_bboxs[index][i], 1)
                feati1 = torch.multiply(feat_all[index, :, :, :], torch.unsqueeze(torch.from_numpy(bi).cuda(), dim=0))
                feati2 = torch.multiply(feat_mov[index, :, :, :], torch.unsqueeze(torch.from_numpy(bi).cuda(), dim=0))
                losses_seg += (torch.mean(0.5 + 0.5 * torch.cosine_similarity(feati1, feati2, dim=1)))
                # losses_seg += torch.sum(1-torch.cosine_similarity(feati1, feati2, dim=0))/torch.sum(torch.from_numpy(bi).cuda())
                sum_seg += 1
            # for j in range(len(mov_bboxs[index])):
            #     bj = np.zeros((256, 256), dtype=np.uint8)
            #     bj = cv2.fillPoly(bj, mov_bboxs[index][j], 1)
            #     featj1 = torch.multiply(feat_all[index, :, :, :], torch.unsqueeze(torch.from_numpy(bj).cuda(), dim=0))
            #     featj2 = torch.multiply(feat_mov[index, :, :, :], torch.unsqueeze(torch.from_numpy(bj).cuda(), dim=0))
            #     losses_seg += (torch.mean(0.5 - 0.5*torch.cosine_similarity(featj1, featj2)))
            #     sum_seg += 1

        return losses_seg / sum_seg if sum_seg != 0 else 0.0


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        # self.check_type_forward((x0, x1, y))

        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.mean(loss)
        return loss


class binary_cross_logits(nn.Module):
    def __init__(self, batch=True):
        super(binary_cross_logits, self).__init__()
        self.batch = batch

    def __call__(self, y_true, y_pred):
        # a = F.binary_cross_entropy_with_logits(y_pred, y_true)
        a = F.binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=torch.Tensor([2.5]).cuda())
        return a


class dice_bce_loss_with_logits1(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss_with_logits1, self).__init__()
        self.batch = batch
        # self.bce_loss = nn.BCELoss()
        # self.bce_loss = F.binary_cross_entropy_with_logits()

    def soft_dice_coeff(self, y_true, y_pred):
        y_pred = torch.sigmoid(y_pred)
        smooth = 0.0  # may change
        # smooth = 1.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        # score = (2. * intersection + smooth) / (i + j + smooth)
        score = (intersection + smooth) / (i + j - intersection + smooth)  # iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred, feat1, feat2):
        f_lab = 1 - y_true
        dis = torch.pairwise_distance(feat1, feat2)
        c = torch.sum(torch.multiply(f_lab, dis)) / torch.sum(f_lab)

        a = F.binary_cross_entropy_with_logits(y_pred, y_true, weight=torch.Tensor([0.7]).cuda())

        # a = F.binary_cross_entropy_with_logits(y_pred, y_true)
        # a = nn.BCEWithLogitsLoss(y_pred, y_true, pos_weight=torch.Tensor([0.7]).cuda())
        # b = self.bce_loss(y_pred, y_true)

        # return a
        b = self.soft_dice_loss(y_true, y_pred)
        # return b + 4*a
        return a + b + c


class triple_dis_loss(nn.Module):
    def __init__(self):
        super(triple_dis_loss, self).__init__()

    def __call__(self, label, feat1, feat2):
        t_lab = label
        f_lab = 1 - label
        dis = torch.pairwise_distance(feat1, feat2)
        loss = torch.sum(torch.multiply(f_lab, dis)) / torch.sum(f_lab)
        # loss = max(torch.sum(torch.multiply(f_lab,dis))/torch.sum(f_lab)-torch.sum(torch.multiply(t_lab,dis))/torch.sum(t_lab)+0.8,0)
        return loss


class dice_bce_loss_with_logits_instance(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss_with_logits_instance, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_true, y_pred):
        y_pred = torch.sigmoid(y_pred)

        smooth = 0.0  # may change
        # smooth = 1.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        # score = (2. * intersection + smooth) / (i + j + smooth)
        score = (intersection + smooth) / (i + j - intersection + smooth)  # iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred, ann):

        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred, ann):
        a = F.binary_cross_entropy_with_logits(y_pred, y_true)
        y_pred = torch.sigmoid(y_pred)
        losses = []
        an = ann.cpu().numpy()
        pred = y_pred.cpu().detach().numpy()
        for img in range(4):
            im = an[img, :, :]
            p = pred[img, :, :]
            na = np.max(im)
            for i in range(int(na)):
                aera = np.zeros((512, 512))
                aera[im == i + 1] = 1
                losses.append((1 - np.sum(aera * p) / (np.sum(aera) + 0.1)))
            aera = np.zeros((512, 512))
            aera[im == 0] = 1
            losses.append(np.sum(aera * p) / (np.sum(aera) + 0.1))
        if len(losses) > 0:
            b = np.mean(losses)
            b = torch.Tensor([b]).cuda()
        else:
            b = a
        return 2 * a + b


class texture_loss_with_logits_instance(nn.Module):
    def __init__(self, batch=True):
        super(texture_loss_with_logits_instance, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_true, y_pred):
        y_pred = torch.sigmoid(y_pred)

        smooth = 0.0  # may change
        # smooth = 1.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        # score = (2. * intersection + smooth) / (i + j + smooth)
        score = (intersection + smooth) / (i + j - intersection + smooth)  # iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, gray, y_true, y_pred, ann):
        a = self.soft_dice_loss(y_true, y_pred)
        y_pred = torch.sigmoid(y_pred)
        losses = []
        an = ann.cpu().numpy()
        pred = y_pred.cpu().detach().numpy()
        imgg = gray.cpu().detach().numpy()
        for img in range(4):
            im = an[img, :, :]
            p = pred[img, :, :]
            g = imgg[img, :, :]
            na = np.max(im)
            back = np.zeros((512, 512))
            back[im == 0] = 1
            bk_avg = (np.sum(back * p * g) / (np.sum(back) + 0.1))
            for i in range(int(na)):
                aera = np.zeros((512, 512))
                aera[im == i + 1] = 1
                bui_avg = (np.sum(aera * p * g) / (np.sum(aera) + 0.1))
                losses.append(min(bk_avg, bui_avg) / max(bk_avg, bui_avg))
        if len(losses) > 0:
            b = np.mean(losses)
            b = torch.Tensor([b]).cuda()
        else:
            b = a
        return a + 2 * b


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[1]


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)  # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore:  # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)]  # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------
def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                    for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# ---------------------------- MULTICLASS LOSSES ---------------------------
def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# ----------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss

