import cv2
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from utils.data_aug_crack import rgb_2_grad3
from skimage.morphology import skeletonize

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_REFLECT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask


def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

    return image, mask


def grade(img):
    x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    mi=np.min(dst)
    ma=np.max(dst)
    res=(dst-mi)/(0.000000001+(ma-mi))
    res[np.isnan(res)]=0
    return res

def truncated_linear_stretch(image, truncated_value=0.5, max_out=255, min_out=0, back_ignore=True):
    def gray_process(gray, dth, uth, maxout=max_out, minout=min_out):
        truncated_down = np.percentile(gray, truncated_value+dth)#0.5-100
        truncated_up = np.percentile(gray, 100 - truncated_value-uth)#0-99.5
        # gray_new = (gray - truncated_down) / (truncated_up - truncated_down) * (maxout - minout) + minout # 分母可能为0
        gray_new = (gray - truncated_down) / (truncated_up - truncated_down) * (maxout - minout) + 0.00001
        gray_new[gray_new < minout] = minout
        gray_new[gray_new > maxout] = maxout
        return np.uint8(gray_new)
    # ignore the background values of 0 and 255
    back_dth = 0
    back_uth = 0
    if back_ignore:
        back_dth = (100-truncated_value)*np.sum(image[:, :, 0] == min_out) / (image.shape[0] * image.shape[1])
        back_uth = (100-truncated_value)*np.sum(image[:, :, 0] == max_out) / (image.shape[0] * image.shape[1])

    if image.shape[2] == 4:
        (b, g, r, l) = cv2.split(image)
        b = gray_process(b,back_dth,back_uth)
        g = gray_process(g,back_dth,back_uth)
        r = gray_process(r,back_dth,back_uth)
        l = gray_process(l,back_dth,back_uth)
        result = cv2.merge((b, g, r, l))
        return result
    else:
        (b, g, r) = cv2.split(image)
        b = gray_process(b,back_dth,back_uth)
        g = gray_process(g,back_dth,back_uth)
        r = gray_process(r,back_dth,back_uth)
        result = cv2.merge((b, g, r))
        return result

def normalize_to_255(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized = (image - min_val) / (max_val - min_val)
    scaled = (normalized * 255).astype(np.uint8)
    return scaled


# def rgb2grad(gray):
#     sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#     sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
#     gradient = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
#     gradient_direction = np.arctan2(sobel_y, sobel_x)
#     enhanced_magnitude = gradient * (np.abs(gradient_direction) > np.pi / 4)
#     enhanced_magnitude = normalize_to_255(enhanced_magnitude)
#
#     log_grad = cv2.GaussianBlur(gray, (3, 3), 0)
#     log_edges = cv2.Laplacian(log_grad, cv2.CV_64F)
#     log_edges = normalize_to_255(log_edges)
#
#     merge_res = np.stack((gray,enhanced_magnitude, log_edges), axis=2)
#     return merge_res
#     # return cv2.merge([gray,enhanced_magnitude, log_edges],)


# def rgb2grad(gray):
#     equalized_image = cv2.equalizeHist(gray)
#     denoised_image1 = cv2.fastNlMeansDenoising(equalized_image, None, h=10, templateWindowSize=7, searchWindowSize=21)
#     sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#     sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
#     gradient = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
#     gradient_direction = np.arctan2(sobel_y, sobel_x)
#     enhanced_magnitude = gradient * (np.abs(gradient_direction) > np.pi / 4)
#     enhanced_magnitude = normalize_to_255(enhanced_magnitude)
#
#     log_grad = cv2.GaussianBlur(gray, (3, 3), 0)
#     log_edges = cv2.Laplacian(log_grad, cv2.CV_64F)
#     log_edges = normalize_to_255(log_edges)
#
#     merge_res = np.stack((gray,enhanced_magnitude, log_edges,normalize_to_255(equalized_image),normalize_to_255(denoised_image1)), axis=2)
#     return merge_res
#     # return cv2.merge([gray,enhanced_magnitude, log_edges],)


def rgb2grad(gray):
    equalized_image = cv2.equalizeHist(gray)
    denoised_image1 = cv2.fastNlMeansDenoising(equalized_image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    gradient_direction = np.arctan2(sobel_y, sobel_x)
    enhanced_magnitude = gradient * (np.abs(gradient_direction) > np.pi / 4)
    # enhanced_magnitude = normalize_to_255(enhanced_magnitude)

    log_grad = cv2.GaussianBlur(gray, (3, 3), 0)
    log_edges = cv2.Laplacian(log_grad, cv2.CV_64F)
    # log_edges = normalize_to_255(log_edges)

    merge_res = np.stack((gray,enhanced_magnitude, log_edges, equalized_image, denoised_image1), axis=2)
    return merge_res


# class Mydataset(Dataset):
#     def __init__(self, path, augment=False, transformrgb=None, transformedge=None, target_transform=None):
#
#         self.aug = augment
#         self.file_path = os.path.dirname(path)
#         self.img_size = 384
#         data = pd.read_csv(path)  # 获取csv表中的数据
#         imgs = []
#         for i in range(len(data)):
#             imgs.append((data.iloc[i, 0], data.iloc[i, 1]))
#         self.imgs = imgs
#         self.transform_rgb = transformrgb
#         self.transform_dge = transformedge
#         self.target_transform = target_transform
#
#     def __getitem__(self, item):
#         if self.aug == False:
#             fn, lab = self.imgs[item]
#             # fn = os.path.join(self.file_path, "image_A/" + fn)
#             # label = os.path.join(self.file_path, "image_A/" + lab)
#             fn = os.path.join(self.file_path, "images/" + fn)
#             label = os.path.join(self.file_path, "masks/" + lab.replace('jpg', 'png'))
#
#             bgr_img = cv2.imread(fn, -1)
#             gt = cv2.imread(label, -1) // 255
#
#             if len(bgr_img.shape)<3:
#                 bgr_img = cv2.merge((bgr_img, bgr_img, bgr_img))
#             bgr_img = bgr_img[:,:,0:3]
#
#             (w, h, c) = bgr_img.shape
#             if w > 256 or h > 256:
#                 sw = (w - 256) // 2
#                 sh = (h - 256) // 2
#                 bgr_img = bgr_img[sw:sw + 256, sh:sh + 256, :]
#                 gt = gt[sw:sw + 256, sh:sh + 256]
#
#             # rgb_img = bgr_img[..., ::-1]  # bgr2rgb
#             rgb_img = cv2.resize(bgr_img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
#             rgb_img = truncated_linear_stretch(rgb_img, truncated_value=0.5, max_out=255, min_out=0, back_ignore=True)
#
#             gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
#             # grad = (255 * grade(gray)).astype(np.uint8)
#             # grad_img = rgb2grad(gray)
#             grad_img = rgb_2_grad3(gray)
#
#             # img = Image.open(fn).convert('RGB')
#             # img = cv2.merge([rgb_img, grad])
#             img = Image.fromarray(rgb_img)
#             if self.transform_rgb is not None:
#                 img = self.transform_rgb(img)
#
#             # grad_img = grad_img.transpose(1,2,0).astype(np.float32)
#             if self.transform_dge is not None:
#                 grad_img = self.transform_dge(grad_img)
#
#             # grad_img=grad_img.transpose(2, 0, 1).astype(np.float32)
#             # grad_img = torch.tensor(grad_img/255.0).float()
#
#             # gt = cv2.imread(label, -1)
#             gt = cv2.resize(gt, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
#             return img, grad_img, gt, lab
#
#         else:
#             # 进行数据增强
#             fn, lab = self.imgs[item]
#             # train with data.cvs
#             fn = os.path.join(self.file_path, "images/" + fn)
#             label = os.path.join(self.file_path, "masks/" + lab.replace('jpg', 'png'))
#
#             gt = cv2.imread(label, -1) // 255
#             # gt = cv2.imread(label, -1)
#             image = cv2.imread(fn, -1)
#
#             if len(image.shape)<3:
#                 image = cv2.merge((image, image, image))
#             image = image[:,:,0:3]
#
#             (w, h, c) = image.shape
#
#             if w>256 or h>256:
#                 sw = (w-256)//2
#                 sh = (h - 256) // 2
#                 image = image[sw:sw+256, sh:sh+256, :]
#                 gt = gt[sw:sw + 256, sh:sh + 256]
#
#             image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
#             gt = cv2.resize(gt, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
#
#             image = randomHueSaturationValue(image,
#                                              hue_shift_limit=(-20, 20),
#                                              sat_shift_limit=(-25, 25),
#                                              val_shift_limit=(-15, 15))
#
#             image, gt = randomShiftScaleRotate(image, gt,
#                                                shift_limit=(-0.1, 0.1),
#                                                scale_limit=(-0.5, 0.5),
#                                                aspect_limit=(-0.1, 0.1),
#                                                rotate_limit=(-10, 10))
#
#             image, gt = randomHorizontalFlip(image, gt)
#             image, gt = randomVerticleFlip(image, gt)
#             image, gt = randomRotate90(image, gt)
#
#             # rgb_img = image[..., ::-1]  # bgr2rgb
#             image = truncated_linear_stretch(image, truncated_value=0.5, max_out=255, min_out=0, back_ignore=True)
#
#             gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#             # grad = (255 * grade(gray)).astype(np.uint8)
#             grad_img = rgb_2_grad3(gray)
#             # grad_img = rgb2grad(gray)
#
#             # img = cv2.merge([rgb_img, grad])
#             img = Image.fromarray(image)
#             if self.transform_rgb is not None:
#                 img = self.transform_rgb(img.copy())
#
#             # grad_img=grad_img.transpose(2, 0, 1).astype(np.float32)
#             # grad_img = torch.tensor(grad_img / 255.0).float()
#
#             # grad_img = grad_img.transpose(1,2,0).astype(np.float32)
#             # grad_img = Image.fromarray(grad_img)
#             if self.transform_dge is not None:
#                 grad_img = self.transform_dge(grad_img)
#
#             return img, grad_img, gt.copy(), lab
#
#     def __len__(self):
#         return len(self.imgs)


# best-dataload
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
            # rgb_img = cv2.resize(rgb_img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            # bgr_img = truncated_linear_stretch(rgb_img, truncated_value=0.5, max_out=255, min_out=0, back_ignore=True)
            gt = cv2.imread(label, -1) // 255
            # gt = cv2.resize(gt, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

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
            # image = cv2.resize(image,(self.img_size,self.img_size),interpolation=cv2.INTER_LINEAR)
            # gt = cv2.resize(gt, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)


            # if len(image.shape) < 3:
            #     image = cv2.merge((image, image, image))
            # image = image[:,:,0:3]



            # image = randomHueSaturationValue(image,
            #                                  hue_shift_limit=(-10, 10),
            #                                  sat_shift_limit=(-5, 5),
            #                                  val_shift_limit=(-25, 25))
            #
            # image, gt = randomShiftScaleRotate(image, gt,
            #                                    shift_limit=(-0.1, 0.1),
            #                                    scale_limit=(-0.25, 0.75),
            #                                    aspect_limit=(-0.1, 0.1),
            #                                    rotate_limit=(-20, 20))
            image = randomHueSaturationValue(image,
                                             hue_shift_limit=(-20, 20),
                                             sat_shift_limit=(-25, 25),
                                             val_shift_limit=(-15, 15))

            image, gt = randomShiftScaleRotate(image, gt,
                                               shift_limit=(-0.1, 0.1),
                                               scale_limit=(-0.5, 0.5),
                                               aspect_limit=(-0.1, 0.1),
                                               rotate_limit=(-10, 10))

            image, gt = randomHorizontalFlip(image, gt)
            image, gt = randomVerticleFlip(image, gt)
            image, gt = randomRotate90(image, gt)

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

def random_resize(image,gt):
    w = image.shape[0]
    h = image.shape[1]
    dw = int(image.shape[0]*0.2)
    dh = int(image.shape[1]*0.2)

    x = np.random.randint(-dw, dw)
    y = np.random.randint(-dh, dh)
    if x < 0:
        if y < 0:
            image = image[0:x, 0:y, :]
            gt = gt[0:x, 0:y]
        else:
            image = image[0:x, y:h, :]
            gt = gt[0:x, y:h]
    else:
        if y < 0:
            image = image[x:w, 0:y, :]
            gt = gt[x:w, 0:y]
        else:
            image = image[x:w, y:h, :]
            gt = gt[x:w, y:h]
    return image,gt

class Mydataset2025(Dataset):
    def __init__(self, path,augment=False,transformrgb=None, transformedge=None, target_transform=None):

        self.aug=augment
        self.file_path=os.path.dirname(path)
        self.img_size=512 #448
        data = pd.read_csv(path)  # 获取csv表中的数据
        imgs = []
        for i in range(len(data)):
            imgs.append((data.iloc[i,0], data.iloc[i,1]))
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
            # rgb_img = bgr_img[..., ::-1]  # bgr2rgb

            rgb_img = truncated_linear_stretch(bgr_img, truncated_value=0.5, max_out=255, min_out=0, back_ignore=True)

            rgb_img = cv2.resize(rgb_img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            grad_img = rgb_2_grad3(gray)

            img = Image.fromarray(rgb_img)
            if self.transform_rgb is not None:
                img = self.transform_rgb(img)

            if self.transform_dge is not None:
                grad_img = self.transform_dge(grad_img)

            gt = cv2.imread(label, -1)//255
            gt = cv2.resize(gt, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            return img, grad_img, gt, lab

        else:
            # 进行数据增强
            fn, lab = self.imgs[item]
            # train with data.cvs
            fn = os.path.join(self.file_path, "images/"+ fn)
            label = os.path.join(self.file_path, "masks/"+ lab)

            image = cv2.imread(fn, -1)
            gt = cv2.imread(label, -1)//255

            image = randomHueSaturationValue(image,
                                             hue_shift_limit=(-10, 10),
                                             sat_shift_limit=(-5, 5),
                                             val_shift_limit=(-25, 25))

            image, gt = randomShiftScaleRotate(image, gt,
                                               shift_limit=(-0.1, 0.1),
                                               scale_limit=(-0.25, 0.75),
                                               aspect_limit=(-0.1, 0.1),
                                               rotate_limit=(-20, 20))

            image, gt = randomHorizontalFlip(image, gt)
            image, gt = randomVerticleFlip(image, gt)
            image, gt = randomRotate90(image, gt)

            # rgb_img = image[..., ::-1]  # bgr2rgb
            image, gt = random_resize(image, gt)

            image = truncated_linear_stretch(image, truncated_value=0.5, max_out=255, min_out=0, back_ignore=True)

            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            grad_img = rgb_2_grad3(gray)

            gt = cv2.resize(gt, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

            img = Image.fromarray(image)
            if self.transform_rgb is not None:
                img = self.transform_rgb(img.copy())

            if self.transform_dge is not None:
                grad_img = self.transform_dge(grad_img)

            return img, grad_img, gt.copy(), lab

    def __len__(self):
        return len(self.imgs)


class CrackDataset(Dataset):
    def __init__(self, path, augment=False, transform=None):
        self.file_path = os.path.dirname(path)
        self.img_size = 384
        self.aug = augment
        data = pd.read_csv(path)  # 获取csv表中的数据
        imgs = []
        for i in range(len(data)):
            imgs.append((data.iloc[i, 0], data.iloc[i, 1]))
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, item):
        if self.aug == False:
            fn, lab = self.imgs[item]
            fn = os.path.join(self.file_path, "images/" + fn)
            rgb_img = cv2.imread(fn, -1)
            rgb_img = cv2.resize(rgb_img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

            label = os.path.join(self.file_path, "masks/" + lab)
            gt = cv2.imread(label, -1) // 255
            gt = cv2.resize(gt, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

            gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            grad_img = rgb2grad(gray)

            rgb_img = truncated_linear_stretch(rgb_img, truncated_value=0.5, max_out=255, min_out=0, back_ignore=True)
            img = Image.fromarray(rgb_img)
            if self.transform is not None:
                img = self.transform(img)

            grad_img = Image.fromarray(grad_img)
            if self.transform is not None:
                grad_img = self.transform(grad_img)

            return img, grad_img, gt, lab

        else:
            # 进行数据增强
            fn, lab = self.imgs[item]
            fn = os.path.join(self.file_path, "images/" + fn)
            label = os.path.join(self.file_path, "masks/" + lab)

            image = cv2.imread(fn, -1)
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            gt = cv2.imread(label, -1) // 255
            gt = cv2.resize(gt, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

            image, gt = randomHorizontalFlip(image, gt)
            image, gt = randomVerticleFlip(image, gt)
            image, gt = randomRotate90(image, gt)


            image, gt = randomShiftScaleRotate(image, gt,
                                               shift_limit=(-0.1, 0.1),
                                               scale_limit=(-0.75, 0.75),
                                               aspect_limit=(-0.1, 0.1),
                                               rotate_limit=(-45, 45))

            rgb_img = randomHueSaturationValue(image,
                                             hue_shift_limit=(-20, 20),
                                             sat_shift_limit=(-25, 25),
                                             val_shift_limit=(-25, 25))

            # rgb_img = truncated_linear_stretch(rgb_img, truncated_value=0.5, max_out=255, min_out=0, back_ignore=True)

            gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            grad_img = rgb2grad(gray)

            # img = cv2.merge([rgb_img, grad])
            img = Image.fromarray(rgb_img)
            if self.transform is not None:
                img = self.transform(img.copy())

            grad_img = Image.fromarray(grad_img)
            if self.transform is not None:
                grad_img = self.transform(grad_img)

            return img, grad_img, gt.copy(), lab

    def __len__(self):
        return len(self.imgs)