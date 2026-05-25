import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F

from models import hrnet_w18
from models.senet import SEModule
from functools import partial
from utils.msca_att import MSCAAttention

nonlinearity = partial(F.relu, inplace=True)
# from inplace_abn import InPlaceABN, InPlaceABNSync
# from models.regnet import regnety_064, regnety_008
from models.regnet import regnety_064, regnety_008 #508//564//uu-595--ep4:6237
from timm.models.regnet import regnetz_040 #35
from timm.models.efficientformer_v2 import efficientformerv2_s2
from timm.models.resnest import resnest26d
from timm.models.convnext import convnext_tiny
from timm.models.edgenext import edgenext_small #42
# from timm.models.edgenext import edgenext


from models.hrnet import hrnet_w32
class Dblock_more_dilate(nn.Module):
    def __init__(self, channel):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out


class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
        return out
#
# class Dblock(nn.Module):
#     def __init__(self, channel):
#         super(Dblock, self).__init__()
#         self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
#         self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
#         self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
#         self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
#         # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#
#     def forward(self, x):
#         dilate1_out = nonlinearity(self.dilate1(x))
#         dilate2_out = nonlinearity(self.dilate2(dilate1_out))
#         dilate3_out = nonlinearity(self.dilate3(dilate2_out))
#         dilate4_out = nonlinearity(self.dilate4(dilate3_out))
#         # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
#         out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
#         return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        res = self.conv3(x)
        x = self.norm3(res)
        x = self.relu3(x)
        return x, res


class DinkNet34_less_pool(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet34_less_pool, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        self.dblock = Dblock_more_dilate(256)

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        # Center
        e3 = self.dblock(e3)

        # Decoder
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class DinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(DinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv0 = nn.Conv2d(inplanes, planes, kernel_size=1, padding=0)
        self.bn0 = nn.BatchNorm2d(planes)

        self.conv1 = nn.Conv2d(inplanes, planes,kernel_size=1,padding=0)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.stride = stride
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        residual = self.conv0(residual)
        residual = self.bn0(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class DeepDinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(DeepDinkNet34, self).__init__()

        filters = [128, 256, 512]
        resnet = models.resnet50(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        # self.basicblock4 = BasicBlock(512,512)
        self.basicblock3 = BasicBlock(1024, 512)
        self.basicblock20 = BasicBlock(512, 256)
        self.basicblock21 = BasicBlock(256, 512)
        self.basicblock10 = BasicBlock(256, 128)
        self.basicblock11 = BasicBlock(128, 128)
        self.basicblock12 = BasicBlock(128, 256)
        self.dblock = Dblock(512)

        self.merge_conv2 = nn.Conv2d(768, 256, 1, padding=0)
        self.merge_relu2 = nonlinearity
        self.merge_conv1 = nn.Conv2d(384, 128, 1, padding=0)
        self.merge_relu1 = nonlinearity

        # self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)
        e1 = self.basicblock10(e1)
        e1 = self.basicblock11(e1)
        e1 = self.basicblock12(e1)
        e2 = self.encoder2(e1)
        e2 = self.basicblock20(e2)
        e2 = self.basicblock21(e2)
        e3 = self.encoder3(e2)
        e3 = self.basicblock3(e3)
        # e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e3)

        d3 = self.decoder3(e4)
        d3m = self.merge_conv2(torch.cat((d3,e2),dim=1))
        d3m = self.merge_relu2(d3m)
        d2 = self.decoder2(d3m)
        d2m = self.merge_conv1(torch.cat((d2, e1), dim=1))
        d2m = self.merge_relu1(d2m)
        d1 = self.decoder1(d2m)

        # d3 = self.decoder3(e4) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class DeepDinkNet50(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(DeepDinkNet50, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet101(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.basicblock4 = BasicBlock(1024,256)
        # self.basicblock3 = BasicBlock(1024, 256)
        self.basicblock3 = BasicBlock(1024, 256)
        self.basicblock20 = BasicBlock(512, 256)
        self.basicblock21 = BasicBlock(256, 128)
        self.basicblock10 = BasicBlock(256, 128)
        self.basicblock11 = BasicBlock(128, 64)
        self.basicblock12 = BasicBlock(64, 64)
        self.dblock = Dblock(1024)

        self.merge_conv2 = nn.Conv2d(256, 128, 1, padding=0)
        self.merge_relu2 = nonlinearity
        self.merge_conv1 = nn.Conv2d(128, 64, 1, padding=0)
        self.merge_relu1 = nonlinearity

        # self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x) # 256
        e2 = self.encoder2(e1) # 512
        e3 = self.encoder3(e2) # 1024

        p1 = self.basicblock10(e1)
        p1 = self.basicblock11(p1)
        p1 = self.basicblock12(p1) # 64
        p2 = self.basicblock20(e2)
        p2 = self.basicblock21(p2) # 128
        # p3 = self.basicblock3(e3) # 256

        # Center
        db = self.dblock(e3)
        db = self.basicblock4(db)

        d3 = self.decoder3(db) #128
        d3m = self.merge_conv2(torch.cat((d3,p2),dim=1))
        d3m = self.merge_relu2(d3m)
        d2 = self.decoder2(d3m)
        d2m = self.merge_conv1(torch.cat((d2, p1), dim=1))
        d2m = self.merge_relu1(d2m)
        d1 = self.decoder1(d2m)

        # d3 = self.decoder3(e4) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class OurDinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(OurDinkNet34, self).__init__()

        filters = [256, 512, 1024,2048]
        resnet = models.resnet50(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(1024)

        self.sample_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1,padding=1))
        # self.merge_conv3 = nn.Conv2d(1024, 512, 1, padding=0)
        # self.merge_relu3 = nonlinearity
        # self.merge_conv2 = nn.Conv2d(512, 256, 1, padding=0)
        # self.merge_relu2 = nonlinearity
        # self.merge_conv1 = nn.Conv2d(256, 128, 1, padding=0)
        # self.merge_relu1 = nonlinearity

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        # e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e3)

        # Decoder
        # d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(e4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        res = self.finalconv3(out)
        refine = self.sample_conv(out)
        return F.sigmoid(res), F.sigmoid(refine)


class ASPPModule(nn.Module):

    def __init__(self, features, inner_features=256, out_features=512, dilations=(6, 12, 24)):
        super(ASPPModule, self).__init__()

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1,
                                             bias=False),
                                   nn.BatchNorm2d(inner_features))
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(inner_features))
        self.conv3 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            nn.BatchNorm2d(inner_features))
        self.conv4 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(inner_features))
        self.conv5 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(inner_features))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        bottle = self.bottleneck(out)
        return bottle

class AsppDinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(AsppDinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # self.dblock = Dblock(512)
        self.aspp = ASPPModule(512,512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.aspp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

class CatDinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(CatDinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(512)

        self.merge_conv3 = nn.Conv2d(1024, 512, 1, padding=0)
        self.merge_relu3 = nonlinearity
        self.merge_conv2 = nn.Conv2d(512, 256, 1, padding=0)
        self.merge_relu2 = nonlinearity
        self.merge_conv1 = nn.Conv2d(256, 128, 1, padding=0)
        self.merge_relu1 = nonlinearity

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        ed = self.dblock(e4)

        # Decoder
        d4 = self.merge_conv3(torch.cat((e4, ed), dim=1))
        d4 = self.merge_relu3(d4)
        d4 = self.decoder4(d4)

        d3 = self.merge_conv2(torch.cat((e3, d4), dim=1))
        d3 = self.merge_relu2(d3)
        d3 = self.decoder3(d3)

        d2 = self.merge_conv1(torch.cat((e2, d3), dim=1))
        d2 = self.merge_relu1(d2)
        d2 = self.decoder2(d2)

        d1 = self.decoder1(d2)


        # d4 = self.decoder4(e4) + e3
        # d3 = self.decoder3(d4) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class PositionAttentionModule(nn.Module):
    """ Position attention module"""
    def __init__(self, in_channels, **kwargs):
        super(PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x
        return out

class ChannelAttentionModule(nn.Module):
    """Channel attention module"""
    def __init__(self, **kwargs):
        super(ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)
        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x
        return out

class AttDinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(AttDinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(512)

        # self.pam1 = PositionAttentionModule(in_channels=64)
        # self.pam2 = PositionAttentionModule(in_channels=128)
        self.pam3 = PositionAttentionModule(in_channels=256)
        self.cam = ChannelAttentionModule()
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x[:,0:3,:,:])
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.cam(e4)
        e4 = self.dblock(e4)

        # Decoder
        e3 = self.pam3(e3)
        d4 = self.decoder4(e4) + e3
        # e2 = self.pam2(e2)
        d3 = self.decoder3(d4) + e2
        # e1 = self.pam1(e1)
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out
        # return F.sigmoid(out)


# class AttRegnet34(nn.Module):
#     def __init__(self, num_classes=1, num_channels=3):
#         super(AttRegnet34, self).__init__()
#
#         # filters = [64, 128, 256, 512]
#         # resnet = models.resnet34(pretrained=True)
#
#         filters = [64, 128, 320, 768]
#         self.resnet = regnety_008(pretrained=True)
#
#         # self.firstconv = resnet.conv1
#         # self.firstbn = resnet.bn1
#         # self.firstrelu = resnet.relu
#         # self.firstmaxpool = resnet.maxpool
#         # self.encoder1 = resnet.layer1
#         # self.encoder2 = resnet.layer2
#         # self.encoder3 = resnet.layer3
#         # self.encoder4 = resnet.layer4
#
#         self.dblock = Dblock(768)
#
#         # self.pam1 = PositionAttentionModule(in_channels=64)
#         # self.pam2 = PositionAttentionModule(in_channels=128)
#         self.pam3 = PositionAttentionModule(in_channels=320)
#         self.cam = ChannelAttentionModule()
#         self.decoder4 = DecoderBlock(filters[3], filters[2])
#         self.decoder3 = DecoderBlock(filters[2], filters[1])
#         self.decoder2 = DecoderBlock(filters[1], filters[0])
#         self.decoder1 = DecoderBlock(filters[0], filters[0])
#
#         self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
#         self.finalrelu1 = nonlinearity
#         self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
#         self.finalrelu2 = nonlinearity
#         self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
#
#     def forward(self, x):
#         # Encoder
#         # x = self.firstconv(x[:,0:3,:,:])
#         # x = self.firstbn(x)
#         # x = self.firstrelu(x)
#         # x = self.firstmaxpool(x)
#         # e1 = self.encoder1(x)
#         # e2 = self.encoder2(e1)
#         # e3 = self.encoder3(e2)
#         # e4 = self.encoder4(e3)
#         e1, e2, e3, e4 = self.resnet(x[:,0:3,:,:])
#         # Center
#         e4 = self.cam(e4)
#         e4 = self.dblock(e4)
#
#         # Decoder
#         e3 = self.pam3(e3)
#         d4 = self.decoder4(e4) + e3
#         # e2 = self.pam2(e2)
#         d3 = self.decoder3(d4) + e2
#         # e1 = self.pam1(e1)
#         d2 = self.decoder2(d3) + e1
#         d1 = self.decoder1(d2)
#
#         out = self.finaldeconv1(d1)
#         out = self.finalrelu1(out)
#         out = self.finalconv2(out)
#         out = self.finalrelu2(out)
#         out = self.finalconv3(out)
#         return out
#         # return F.sigmoid(out)


class Net_Tes(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(Net_Tes, self).__init__()

        # filters = [64, 128, 256, 512]
        # resnet = models.resnet34(pretrained=True)

        filters = [64, 128, 320, 768]
        self.resnet = regnety_008(pretrained=True, features_only=True)
        # filters = [48,96,160,304]
        # self.resnet = edgenext_small(pretrained=True, features_only=True)
        # self.resnet = regnetz_040(pretrained=True, features_only=True)
        # filters = [96, 192, 384, 768]
        # self.resnet = convnext_tiny(pretrained=True, features_only=True)
        # self.resnet = resnest26d(pretrained=True, features_only=True)

        # self.dblock = Dblock(768)

        # self.pam1 = PositionAttentionModule(in_channels=64)
        # self.pam2 = PositionAttentionModule(in_channels=128)
        self.pam3 = PositionAttentionModule(in_channels=filters[2])
        self.pam4 = PositionAttentionModule(in_channels=filters[3])
        self.cam = ChannelAttentionModule()
        # self.decoder4 = DecoderBlock(filters[3], filters[2])
        # self.decoder3 = DecoderBlock(filters[2], filters[1])
        # self.decoder2 = DecoderBlock(filters[1], filters[0])
        # self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(2 * filters[2], filters[1])
        self.decoder2 = DecoderBlock(2 * filters[1], filters[0])
        self.decoder1 = DecoderBlock(2 * filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        feat = self.resnet(x[:, 0:3, :, :])
        e1, e2, e3, e4 = feat[1],feat[2],feat[3],feat[4]
        # e1, e2, e3, e4 = self.resnet(x[:, 0:3, :, :])
        # Center

        # e4 = self.dblock(e4)

        # Decoder
        e3 = self.pam3(e3)
        e4 = self.pam4(e4)
        d4 = self.decoder4(e4)
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))

        # e3 = self.pam3(e3)
        # d4 = self.decoder4(e4) + e3
        # d3 = self.decoder3(d4) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)
        d1 = self.cam(d1)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out


class FU_Block(nn.Module):
    def __init__(self, channel):
        super(FU_Block, self).__init__()
        # self.pam = PositionAttentionModule(channel//2)
        self.cam = SEModule(channels=2*channel, reduction=2)
        self.conv_e1 = nn.Conv2d(channel, channel // 2, 1, bias=False)
        self.conv_e2 = nn.Conv2d(2*channel, channel, 1, bias=False)
        self.bne = nn.BatchNorm2d( channel // 2)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nonlinearity

    def forward(self, x1, x2):

        x11 = self.conv_e1(x1)
        # x11 = self.pam(x11)
        x11 = self.bne(x11)
        x21 = self.conv_e1(x2)
        # x21 = self.pam(x21)
        x21 = self.bne(x21)
        out = self.cam(torch.cat((torch.add(x1, x2),x11, x21), dim=1))
        out = self.conv_e2(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class FU_Block2025(nn.Module):
    def __init__(self, ch1,ch2):
        super(FU_Block2025, self).__init__()
        # self.pam = PositionAttentionModule(channel//2)
        self.cam = SEModule(channels=ch1+ch2, reduction=2)
        # self.conv_e1 = nn.Conv2d(channel, channel // 2, 1, bias=False)
        self.conv_e2 = nn.Conv2d(ch1+ch2, ch2, 1, bias=False)
        # self.bne = nn.BatchNorm2d( channel // 2)
        self.bn = nn.BatchNorm2d(ch2)
        self.relu = nonlinearity

    def forward(self, x1, x2):

        # x11 = self.conv_e1(x1)
        # x11 = self.pam(x11)
        # x11 = self.bne(x11)
        # x21 = self.conv_e1(x2)
        # x21 = self.pam(x21)
        # x21 = self.bne(x21)
        out = self.cam(torch.cat((x1, x2), dim=1))
        out = self.conv_e2(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

class FUSION_Block(nn.Module):
    def __init__(self, channel):
        super(FUSION_Block, self).__init__()
        # self.pam = PositionAttentionModule(channel//2)
        self.cam = SEModule(channels=3*channel, reduction=2)
        # self.conv_e1 = nn.Conv2d(channel, channel // 2, 1, bias=False)
        self.conv_e2 = nn.Conv2d(3*channel, channel, 1, bias=False)
        # self.bne = nn.BatchNorm2d( channel // 2)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nonlinearity

    def forward(self, x1, x2):

        # x11 = self.conv_e1(x1)
        # # x11 = self.pam(x11)
        # x11 = self.bne(x11)
        # x21 = self.conv_e1(x2)
        # # x21 = self.pam(x21)
        # x21 = self.bne(x21)
        out = self.cam(torch.cat((torch.add(x1, x2),x1, x2), dim=1))
        out = self.conv_e2(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class UPBlock(nn.Module):
    def __init__(self, in_channels, n_filters=32):
        super(UPBlock, self).__init__()

        self.deconv0 = nn.ConvTranspose2d(in_channels, n_filters // 4, 4, 2, 1)
        self.conv0 = nn.Conv2d(n_filters // 4, n_filters // 4, 3, padding=1)
        self.norm0 = nn.BatchNorm2d(n_filters // 4)
        self.conv1 = nn.Conv2d(n_filters // 4, n_filters//2, 3,padding=1)
        self.norm1 = nn.BatchNorm2d(n_filters//2)
        # self.conv2 = nn.Conv2d(n_filters // 2, n_filters // 2, stride=2, kernel_size=3, padding=1)
        # self.norm2 = nn.BatchNorm2d(n_filters // 2)
        self.down1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(n_filters//2, n_filters//2, 3,padding=1)
        self.norm3 = nn.BatchNorm2d(n_filters//2)

        # self.conv4 = nn.Conv2d(n_filters//2, n_filters, stride=2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(n_filters // 2, n_filters, 3, padding=1)
        self.norm4 = nn.BatchNorm2d(n_filters)
        self.down2 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.deconv0(x)
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.conv1(x)
        x = self.norm1(x)
        # x = self.conv2(x)
        # x0 = self.norm2(x)
        x0 = self.down1(x)

        x1 = self.conv3(x0)
        x1 = self.norm3(x1)
        x1 = self.conv4(x1)
        x1 = self.norm4(x1)
        x1 = self.down2(x1)

        return x0, x1


class UPBlock2(nn.Module):
    def __init__(self, in_channels, n_filters=32, n_filters2=32):
        super(UPBlock2, self).__init__()

        self.deconv0 = nn.ConvTranspose2d(in_channels, n_filters // 2, 4, 2, 1)
        self.conv0 = nn.Conv2d(n_filters // 2, n_filters // 2, 3, padding=1)
        self.norm0 = nn.BatchNorm2d(n_filters // 2)
        self.conv1 = nn.Conv2d(n_filters // 2, n_filters//2, 3,padding=1)
        self.norm1 = nn.BatchNorm2d(n_filters//2)
        self.down1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(n_filters//2, n_filters//2, 3,padding=1)
        self.norm3 = nn.BatchNorm2d(n_filters//2)
        self.conv4 = nn.Conv2d(n_filters // 2, n_filters, 3, padding=1)
        self.norm4 = nn.BatchNorm2d(n_filters)
        self.down2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(n_filters, n_filters, 3, padding=1)
        self.norm5 = nn.BatchNorm2d(n_filters)
        self.conv6 = nn.Conv2d(n_filters, n_filters2, 3, padding=1)
        self.norm6 = nn.BatchNorm2d(n_filters2)
        self.down3 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.deconv0(x)
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x0 = self.down1(x)

        x1 = self.conv3(x0)
        x1 = self.norm3(x1)
        x1 = self.conv4(x1)
        x1 = self.norm4(x1)
        x1 = self.down2(x1)

        x2 = self.conv5(x1)
        x2 = self.norm5(x2)
        x2 = self.conv6(x2)
        x2 = self.norm6(x2)
        x2 = self.down3(x2)

        return x0, x1, x2

class UPBlock2026(nn.Module):
    def __init__(self, in_channels, n_filters=32, n_filters2=32):
        super(UPBlock2026, self).__init__()
        self.cam = ChannelAttentionModule()
        self.deconv0 = nn.ConvTranspose2d(in_channels, n_filters // 2, 4, 2, 1)
        self.conv0 = nn.Conv2d(n_filters // 2, n_filters // 2, 3, padding=1)
        self.norm0 = nn.BatchNorm2d(n_filters // 2)
        self.conv1 = nn.Conv2d(n_filters // 2, n_filters//2, 3,padding=1)
        self.norm1 = nn.BatchNorm2d(n_filters//2)
        self.down1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(n_filters//2, n_filters//2, 3,padding=1)
        self.norm3 = nn.BatchNorm2d(n_filters//2)
        self.conv4 = nn.Conv2d(n_filters, n_filters, 3, padding=1)
        self.norm4 = nn.BatchNorm2d(n_filters)
        self.down2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(n_filters, n_filters, 3, padding=1)
        self.norm5 = nn.BatchNorm2d(n_filters)
        self.conv6 = nn.Conv2d(2*n_filters, n_filters2, 3, padding=1)
        self.norm6 = nn.BatchNorm2d(n_filters2)
        self.down3 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.deconv0(x)
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x0 = self.down1(x)

        x1 = self.conv3(x0)
        x1 = self.norm3(x1)
        x1 = self.conv4(self.cam(torch.cat((x0,x1),dim=1)))
        x1 = self.norm4(x1)
        x1 = self.down2(x1)

        x2 = self.conv5(x1)
        x2 = self.norm5(x2)
        x2 = self.conv6(self.cam(torch.cat((x1,x2),dim=1)))
        x2 = self.norm6(x2)
        x2 = self.down3(x2)

        return x0, x1, x2


class UPBlockatt(nn.Module):
    def __init__(self, in_channels, n_filters=32, n_filters2=32):
        super(UPBlockatt, self).__init__()

        self.deconv0 = nn.ConvTranspose2d(in_channels, n_filters // 2, 4, 2, 1)
        self.conv0 = nn.Conv2d(n_filters // 2, n_filters // 2, 3, padding=1)
        self.norm0 = nn.BatchNorm2d(n_filters // 2)
        self.conv1 = nn.Conv2d(n_filters // 2, n_filters//2, 3,padding=1)
        self.norm1 = nn.BatchNorm2d(n_filters//2)
        self.down1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(n_filters//2, n_filters//2, 3,padding=1)
        self.norm3 = nn.BatchNorm2d(n_filters//2)
        self.conv4 = nn.Conv2d(n_filters // 2, n_filters, 3, padding=1)
        self.norm4 = nn.BatchNorm2d(n_filters)
        self.down2 = nn.MaxPool2d(2, 2)
        self.cam1 = SEModule(channels=(n_filters+n_filters//2),reduction=2)
        self.convatt1 = nn.Conv2d((n_filters+n_filters//2), n_filters, 1)

        self.conv5 = nn.Conv2d(n_filters, n_filters, 3, padding=1)
        self.norm5 = nn.BatchNorm2d(n_filters)
        self.conv6 = nn.Conv2d(n_filters, n_filters2, 3, padding=1)
        self.norm6 = nn.BatchNorm2d(n_filters2)
        self.down3 = nn.MaxPool2d(2, 2)
        self.cam2 = SEModule(channels=(n_filters+n_filters2),reduction=2)
        self.convatt2 = nn.Conv2d((n_filters+n_filters2), n_filters2, 1)

    def forward(self, x):
        x = self.deconv0(x)
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x0 = self.down1(x)

        x1 = self.conv3(x0)
        x1 = self.norm3(x1)
        x1 = self.conv4(x1)
        x1 = self.norm4(x1)

        x1 = self.convatt1(self.cam1(torch.cat([x0,x1], dim=1)))
        x1 = self.down2(x1)

        x2 = self.conv5(x1)
        x2 = self.norm5(x2)
        x2 = self.conv6(x2)
        x2 = self.norm6(x2)
        x2 = self.convatt2(self.cam2(torch.cat([x1, x2], dim=1)))

        x2 = self.down3(x2)

        return x0, x1, x2


class ECBlock(nn.Module):
    def __init__(self, in_channels, n_filters=32):
        super(ECBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, n_filters//2, 3,padding=1)
        self.norm1 = nn.BatchNorm2d(n_filters//2)

        self.conv2 = nn.Conv2d(n_filters//2, n_filters//2, 3,padding=1)
        self.norm2 = nn.BatchNorm2d(n_filters//2)

        self.conv3 = nn.Conv2d(n_filters//2, n_filters, stride=2, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x0 = self.norm2(x)

        x1 = self.conv3(x0)
        x1 = self.norm3(x1)

        return x0, x1


class Net_TesUU(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(Net_TesUU, self).__init__()

        filters = [32, 64, 128, 320, 768]
        self.resnet = regnety_008(pretrained=True, features_only=True)
        self.encod = ECBlock(3,filters[0])
        # self.c4 = DF_Block6(filters[4])
        # self.c3 = DF_Block6(filters[3])
        # self.c2 = DF_Block6(filters[2])
        # self.c1 = DF_Block6(filters[1])
        self.c0 = FU_Block(filters[0])

        # filters = [144, 288, 576, 1296]
        # self.resnet = regnety_064(pretrained=True, features_only=True)
        # filters = [48,96,160,304]
        # self.resnet = edgenext_small(pretrained=True, features_only=True)
        # self.resnet = regnetz_040(pretrained=True, features_only=True)
        # filters = [96, 192, 384, 768]
        # self.resnet = convnext_tiny(pretrained=True, features_only=True)
        # self.resnet = resnest26d(pretrained=True, features_only=True)

        # self.dblock = Dblock(768)

        # self.pam1 = PositionAttentionModule(in_channels=64)
        # self.pam2 = PositionAttentionModule(in_channels=128)
        self.pam3 = PositionAttentionModule(in_channels=filters[3])
        self.pam4 = PositionAttentionModule(in_channels=filters[4])
        self.cam = ChannelAttentionModule()
        # self.decoder4 = DecoderBlock(filters[3], filters[2])
        # self.decoder3 = DecoderBlock(filters[2], filters[1])
        # self.decoder2 = DecoderBlock(filters[1], filters[0])
        # self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.decoder4 = DecoderBlock(filters[4], filters[3])
        self.decoder3 = DecoderBlock(2 * filters[3], filters[2])
        self.decoder2 = DecoderBlock(2 * filters[2], filters[1])
        self.decoder1 = DecoderBlock(2 * filters[1], filters[0])
        self.decoder0 = DecoderBlock(2 * filters[0], filters[0])

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv1 = nn.Conv2d(filters[0], 32, 3, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x, input2):
        # Encoder
        feat1 = self.resnet(x[:, 0:3, :, :])
        e0, e1, e2, e3, e4 = feat1[0],feat1[1],feat1[2],feat1[3],feat1[4]

        # feat2 = self.resnet(input2)
        # e20, e21, e22, e23, e24 = feat2[0], feat2[1], feat2[2], feat2[3], feat2[4]
        e10 = self.encod(input2)
        e0 = self.c0(e0, e10)
        # e1 = self.c1(e11, e21)
        # e2 = self.c2(e12, e22)
        # e3 = self.c3(e13, e23)
        # e4 = self.c4(e14, e24)

        # e4 = self.dblock(e4)

        # Decoder
        e3 = self.pam3(e3)
        e4 = self.pam4(e4)
        d4 = self.decoder4(e4)
        d3 = self.decoder3(self.cam(torch.cat((d4, e3), dim=1)))
        d2 = self.decoder2(self.cam(torch.cat((d3, e2), dim=1)))
        d1 = self.decoder1(self.cam(torch.cat((d2, e1), dim=1)))
        d0 = self.decoder0(self.cam(torch.cat((d1, e0), dim=1)))

        # e3 = self.pam3(e3)
        # d4 = self.decoder4(e4) + e3
        # d3 = self.decoder3(d4) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)
        d0 = self.cam(d0)
        out = self.finaldeconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out


class Net_TesUU2(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(Net_TesUU2, self).__init__()

        filters = [32, 64, 128, 320, 768]
        self.resnet = regnety_008(pretrained=True, features_only=True)
        self.encod = ECBlock(3,filters[0])
        # self.c4 = DF_Block6(filters[4])
        # self.c3 = DF_Block6(filters[3])
        # self.c2 = DF_Block6(filters[2])
        # self.c1 = DF_Block6(filters[1])
        self.c0 = FU_Block(filters[0])

        # filters = [144, 288, 576, 1296]
        # self.resnet = regnety_064(pretrained=True, features_only=True)
        # filters = [48,96,160,304]
        # self.resnet = edgenext_small(pretrained=True, features_only=True)
        # self.resnet = regnetz_040(pretrained=True, features_only=True)
        # filters = [96, 192, 384, 768]
        # self.resnet = convnext_tiny(pretrained=True, features_only=True)
        # self.resnet = resnest26d(pretrained=True, features_only=True)

        # self.dblock = Dblock(768)

        # self.pam1 = PositionAttentionModule(in_channels=64)
        # self.pam2 = PositionAttentionModule(in_channels=128)
        # self.pam3 = PositionAttentionModule(in_channels=filters[3])
        # self.pam4 = PositionAttentionModule(in_channels=filters[4])
        self.cam = ChannelAttentionModule()
        # self.decoder4 = DecoderBlock(filters[3], filters[2])
        # self.decoder3 = DecoderBlock(filters[2], filters[1])
        # self.decoder2 = DecoderBlock(filters[1], filters[0])
        # self.decoder1 = DecoderBlock(filters[0], filters[0])
        # self.decoder4 = DecoderBlock(filters[4], filters[3])
        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(2 * filters[2], filters[1])
        self.decoder1 = DecoderBlock(2 * filters[1], filters[0])
        self.decoder0 = DecoderBlock(2 * filters[0], filters[0])

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv1 = nn.Conv2d(filters[0], 32, 3, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x, input2):
        # Encoder
        feat1 = self.resnet(x[:, 0:3, :, :])
        e0, e1, e2, e3 = feat1[0],feat1[1],feat1[2],feat1[3]

        # feat2 = self.resnet(input2)
        # e20, e21, e22, e23, e24 = feat2[0], feat2[1], feat2[2], feat2[3], feat2[4]
        e10 = self.encod(input2)
        e0 = self.c0(e0, e10)
        # e1 = self.c1(e11, e21)
        # e2 = self.c2(e12, e22)
        # e3 = self.c3(e13, e23)
        # e4 = self.c4(e14, e24)

        # e4 = self.dblock(e4)

        # Decoder
        # e3 = self.pam3(e3)
        # e4 = self.pam4(e4)
        # d4 = self.decoder4(e4)
        d3 = self.decoder3(e3)
        d2 = self.decoder2(self.cam(torch.cat((d3, e2), dim=1)))
        d1 = self.decoder1(self.cam(torch.cat((d2, e1), dim=1)))
        d0 = self.decoder0(self.cam(torch.cat((d1, e0), dim=1)))

        # e3 = self.pam3(e3)
        # d4 = self.decoder4(e4) + e3
        # d3 = self.decoder3(d4) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)
        d0 = self.cam(d0)
        out = self.finaldeconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out


class RegNet_MultiHead(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(RegNet_MultiHead, self).__init__()

        filters = [32, 64, 128, 320, 768]
        self.resnet = regnety_008(pretrained=True, features_only=True)
        self.encod = ECBlock(3,filters[0])
        self.c0 = FU_Block(filters[0])

        self.cam = ChannelAttentionModule()

        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(2 * filters[2], filters[1])
        self.decoder1 = DecoderBlock(2 * filters[1], filters[0])
        self.decoder0 = DecoderBlock(2 * filters[0], filters[0])

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv1 = nn.Conv2d(filters[0]+16, 32, 3, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x, input2):
        # Encoder
        feat1 = self.resnet(x[:, 0:3, :, :])
        e0, e1, e2, e3 = feat1[0],feat1[1],feat1[2],feat1[3]

        # feat2 = self.resnet(input2)
        # e20, e21, e22, e23, e24 = feat2[0], feat2[1], feat2[2], feat2[3], feat2[4]
        e00,e01  = self.encod(input2)
        e0 = self.c0(e0, e01)
        # e1 = self.c1(e11, e21)
        # e2 = self.c2(e12, e22)
        # e3 = self.c3(e13, e23)
        # e4 = self.c4(e14, e24)

        # e4 = self.dblock(e4)

        # Decoder
        # e3 = self.pam3(e3)
        # e4 = self.pam4(e4)
        # d4 = self.decoder4(e4)
        d3 = self.decoder3(e3)
        d2 = self.decoder2(self.cam(torch.cat((d3, e2), dim=1)))
        d1 = self.decoder1(self.cam(torch.cat((d2, e1), dim=1)))
        d0 = self.decoder0(self.cam(torch.cat((d1, e0), dim=1)))

        # e3 = self.pam3(e3)
        # d4 = self.decoder4(e4) + e3
        # d3 = self.decoder3(d4) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)
        d0 = self.cam(torch.cat((d0, e00), dim=1))
        out = self.finaldeconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out


class RegNet_MultiHeadU2(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(RegNet_MultiHeadU2, self).__init__()

        filters = [32, 64, 128, 320, 768]
        self.resnet = regnety_008(pretrained=True, features_only=True)
        self.encod = UPBlock(3,filters[0])

        self.c0 = FU_Block(filters[0])

        self.cam = ChannelAttentionModule()

        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(2 * filters[2], filters[1])
        self.decoder1 = DecoderBlock(2 * filters[1], filters[0])
        self.decoder0 = DecoderBlock(2 * filters[0], filters[0])

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv1 = nn.Conv2d(filters[0]+16, 32, 3, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x, input2):
        # Encoder
        feat1 = self.resnet(x[:, 0:3, :, :])
        e0, e1, e2, e3 = feat1[0],feat1[1],feat1[2],feat1[3]

        # feat2 = self.resnet(input2)
        # e20, e21, e22, e23, e24 = feat2[0], feat2[1], feat2[2], feat2[3], feat2[4]
        e00,e01  = self.encod(input2)
        e0 = self.c0(e0, e01)
        # e1 = self.c1(e11, e21)
        # e2 = self.c2(e12, e22)
        # e3 = self.c3(e13, e23)
        # e4 = self.c4(e14, e24)

        # e4 = self.dblock(e4)

        # Decoder
        # e3 = self.pam3(e3)
        # e4 = self.pam4(e4)
        # d4 = self.decoder4(e4)
        d3 = self.decoder3(e3)
        d2 = self.decoder2(self.cam(torch.cat((d3, e2), dim=1)))
        d1 = self.decoder1(self.cam(torch.cat((d2, e1), dim=1)))
        d0 = self.decoder0(self.cam(torch.cat((d1, e0), dim=1)))

        # e3 = self.pam3(e3)
        # d4 = self.decoder4(e4) + e3
        # d3 = self.decoder3(d4) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)
        d0 = self.cam(torch.cat((d0, e00), dim=1))
        out = self.finaldeconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out


# class RegNet_MultiHeadU3(nn.Module):
#     def __init__(self, num_classes=1, num_channels=3):
#         super(RegNet_MultiHeadU3, self).__init__()
#
#         filters = [32, 64, 128, 320, 768]
#         self.resnet = regnety_008(pretrained=True, features_only=True)
#         self.encod = UPBlock2(10,filters[0])
#
#         self.c0 = FU_Block(filters[0])
#         self.c1 = FU_Block(filters[1])
#
#         self.cam = ChannelAttentionModule()
#
#         self.decoder3 = DecoderBlock(filters[3], filters[2])
#         self.decoder2 = DecoderBlock(2 * filters[2], filters[1])
#         self.decoder1 = DecoderBlock(2 * filters[1], filters[0])
#         self.decoder0 = DecoderBlock(2 * filters[0], filters[0])
#
#         # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
#         self.finaldeconv1 = nn.Conv2d(filters[0]+16, 32, 3, padding=1)
#         self.finalrelu1 = nonlinearity
#         self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
#         self.finalrelu2 = nonlinearity
#         self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
#
#     def forward(self, x, input2):
#         # Encoder
#         feat1 = self.resnet(x[:, 0:3, :, :])
#         e0, e1, e2, e3 = feat1[0],feat1[1],feat1[2],feat1[3]
#
#         # feat2 = self.resnet(input2)
#         # e20, e21, e22, e23, e24 = feat2[0], feat2[1], feat2[2], feat2[3], feat2[4]
#         e00,e01,e02  = self.encod(input2)
#         e0 = self.c0(e0, e01)
#         e1 = self.c1(e1, e02)
#         # e1 = self.c1(e11, e21)
#         # e2 = self.c2(e12, e22)
#         # e3 = self.c3(e13, e23)
#         # e4 = self.c4(e14, e24)
#
#         # e4 = self.dblock(e4)
#
#         # Decoder
#         # e3 = self.pam3(e3)
#         # e4 = self.pam4(e4)
#         # d4 = self.decoder4(e4)
#         d3 = self.decoder3(e3)
#         d2 = self.decoder2(self.cam(torch.cat((d3, e2), dim=1)))
#         d1 = self.decoder1(self.cam(torch.cat((d2, e1), dim=1)))
#         d0 = self.decoder0(self.cam(torch.cat((d1, e0), dim=1)))
#
#         # e3 = self.pam3(e3)
#         # d4 = self.decoder4(e4) + e3
#         # d3 = self.decoder3(d4) + e2
#         # d2 = self.decoder2(d3) + e1
#         # d1 = self.decoder1(d2)
#         d0 = self.cam(torch.cat((d0, e00), dim=1))
#         out = self.finaldeconv1(d0)
#         out = self.finalrelu1(out)
#         out = self.finalconv2(out)
#         out = self.finalrelu2(out)
#         out = self.finalconv3(out)
#         return out

class RegNet_MultiHeadU3(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(RegNet_MultiHeadU3, self).__init__()

        filters = [32, 64, 128, 320, 768]
        self.resnet = regnety_008(pretrained=True, features_only=True)
        self.encod = UPBlock2(10,filters[0])

        self.c0 = FU_Block(filters[0])
        self.c1 = FU_Block(filters[1])

        self.cam = ChannelAttentionModule()

        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(2 * filters[2], filters[1])
        self.decoder1 = DecoderBlock(2 * filters[1], filters[0])
        self.decoder0 = DecoderBlock(2 * filters[0], filters[0])

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv1 = nn.Conv2d(filters[0]+16, 32, 3, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x, input2):
        # Encoder
        feat1 = self.resnet(x[:, 0:3, :, :])
        e0, e1, e2, e3 = feat1[0],feat1[1],feat1[2],feat1[3]

        # feat2 = self.resnet(input2)
        # e20, e21, e22, e23, e24 = feat2[0], feat2[1], feat2[2], feat2[3], feat2[4]
        e00,e01,e02  = self.encod(input2)
        e0 = self.c0(e0, e01)
        e1 = self.c1(e1, e02)
        # e1 = self.c1(e11, e21)
        # e2 = self.c2(e12, e22)
        # e3 = self.c3(e13, e23)
        # e4 = self.c4(e14, e24)

        # e4 = self.dblock(e4)

        # Decoder
        # e3 = self.pam3(e3)
        # e4 = self.pam4(e4)
        # d4 = self.decoder4(e4)
        d3 = self.decoder3(e3)
        d2 = self.decoder2(self.cam(torch.cat((d3, e2), dim=1)))
        d1 = self.decoder1(self.cam(torch.cat((d2, e1), dim=1)))
        d0 = self.decoder0(self.cam(torch.cat((d1, e0), dim=1)))

        # e3 = self.pam3(e3)
        # d4 = self.decoder4(e4) + e3
        # d3 = self.decoder3(d4) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)
        d0 = self.cam(torch.cat((d0, e00), dim=1))
        out = self.finaldeconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out

class RegNet_MultiHeadU3_64(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(RegNet_MultiHeadU3_64, self).__init__()

        # filters = [32, 64, 128, 320, 768]
        filters = [32, 144, 288, 576]
        self.resnet = regnety_064(pretrained=True, features_only=True)
        self.encod = UPBlock2(10,filters[0],filters[1])

        self.c0 = FU_Block(filters[0])
        self.c1 = FU_Block(filters[1])

        self.cam = ChannelAttentionModule()

        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(2 * filters[2], filters[1])
        self.decoder1 = DecoderBlock(2 * filters[1], filters[0])
        self.decoder0 = DecoderBlock(2 * filters[0], filters[0])

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv1 = nn.Conv2d(filters[0]+filters[0]//2, 32, 3, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x, input2):
        # Encoder
        feat1 = self.resnet(x[:, 0:3, :, :])
        e0, e1, e2, e3 = feat1[0],feat1[1],feat1[2],feat1[3]

        # feat2 = self.resnet(input2)
        # e20, e21, e22, e23, e24 = feat2[0], feat2[1], feat2[2], feat2[3], feat2[4]
        e00,e01,e02  = self.encod(input2)
        e0 = self.c0(e0, e01)
        e1 = self.c1(e1, e02)
        # e1 = self.c1(e11, e21)
        # e2 = self.c2(e12, e22)
        # e3 = self.c3(e13, e23)
        # e4 = self.c4(e14, e24)

        # e4 = self.dblock(e4)

        # Decoder
        # e3 = self.pam3(e3)
        # e4 = self.pam4(e4)
        # d4 = self.decoder4(e4)
        d3 = self.decoder3(e3)
        d2 = self.decoder2(self.cam(torch.cat((d3, e2), dim=1)))
        d1 = self.decoder1(self.cam(torch.cat((d2, e1), dim=1)))
        d0 = self.decoder0(self.cam(torch.cat((d1, e0), dim=1)))

        # e3 = self.pam3(e3)
        # d4 = self.decoder4(e4) + e3
        # d3 = self.decoder3(d4) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)
        d0 = self.cam(torch.cat((d0, e00), dim=1))
        out = self.finaldeconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out


class RegNet_MultiHeadU3_2025(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(RegNet_MultiHeadU3_2025, self).__init__()

        # filters = [32, 64, 128, 320, 768]
        spatial=[32, 64]
        filters = [32, 144, 288, 576]
        self.resnet = regnety_064(pretrained=True, features_only=True)
        # self.encod = UPBlock2(10,filters[0],filters[1])
        # self.encod = UPBlock2(10, spatial[0], spatial[1])
        self.encod = UPBlock2026(10, spatial[0], spatial[1])

        self.c0 = FU_Block2025(spatial[0],filters[0])
        self.c1 = FU_Block2025(spatial[1],filters[1])

        self.cam = ChannelAttentionModule()

        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(2 * filters[2], filters[1])
        self.decoder1 = DecoderBlock(2 * filters[1], filters[0])
        self.decoder0 = DecoderBlock(2 * filters[0], filters[0])


        self.final_center = nn.Conv2d(filters[0] + filters[0] // 2, 32, kernel_size=1)
        self.final_ctr_relu1 = nonlinearity
        self.final_ctr_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_ctr_relu2 = nonlinearity
        self.final_ctr_conv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv1 = nn.Conv2d(filters[0]+filters[0]//2, 32, 3, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalconv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x, input2):
        # Encoder
        feat1 = self.resnet(x[:, 0:3, :, :])
        e0, e1, e2, e3 = feat1[0],feat1[1],feat1[2],feat1[3]

        # feat2 = self.resnet(input2)
        # e20, e21, e22, e23, e24 = feat2[0], feat2[1], feat2[2], feat2[3], feat2[4]
        e00,e01,e02  = self.encod(input2)  #16,32,64
        e0 = self.c0(e0, e01)
        e1 = self.c1(e1, e02)
        # e1 = self.c1(e11, e21)
        # e2 = self.c2(e12, e22)
        # e3 = self.c3(e13, e23)
        # e4 = self.c4(e14, e24)

        # e4 = self.dblock(e4)

        # Decoder
        # e3 = self.pam3(e3)
        # e4 = self.pam4(e4)
        # d4 = self.decoder4(e4)
        d3 = self.decoder3(e3)
        d2 = self.decoder2(self.cam(torch.cat((d3, e2), dim=1)))
        d1 = self.decoder1(self.cam(torch.cat((d2, e1), dim=1)))
        d0 = self.decoder0(self.cam(torch.cat((d1, e0), dim=1)))

        # e3 = self.pam3(e3)
        # d4 = self.decoder4(e4) + e3
        # d3 = self.decoder3(d4) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)
        d0 = self.cam(torch.cat((d0, e00), dim=1))

        out = self.finaldeconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out

        # ctr1 = self.final_center(d0)
        # ctr1 = self.final_ctr_relu1(ctr1)
        # ctr1 = self.final_ctr_conv2(ctr1)
        # ctr1 = self.final_ctr_relu2(ctr1)
        # ctr = self.final_ctr_conv3(ctr1)
        #
        # out = self.finaldeconv1(d0)
        # out = self.finalrelu1(out)
        # out = self.finalconv2(torch.add(ctr1,out))
        # out = self.finalrelu2(out)
        # out = self.finalconv3(out)
        # return out, ctr

# class RegNet_MultiHeadU3_Land_Con(nn.Module):
#     def __init__(self, num_classes=1, num_channels=3):
#         super(RegNet_MultiHeadU3_Land_Con, self).__init__()
#
#         # filters = [32, 64, 128, 320, 768]
#         spatial=[32, 64]
#         filters = [32, 144, 288, 576]
#         self.resnet = regnety_064(pretrained=True, features_only=True)
#         # self.encod = UPBlock2(10,filters[0],filters[1])
#         # self.encod = UPBlock2(10, spatial[0], spatial[1])
#         self.encod = UPBlock2026(10, spatial[0], spatial[1])
#
#         self.c0 = FU_Block2025(spatial[0],filters[0])
#         self.c1 = FU_Block2025(spatial[1],filters[1])
#
#         self.pam2 = PositionAttentionModule(in_channels=filters[2])
#         self.pam3 = PositionAttentionModule(in_channels=filters[3])
#         self.dblock = Dblock(filters[3])
#         self.cam = ChannelAttentionModule()
#
#         self.decoder3 = DecoderBlock(filters[3], filters[2])
#         self.decoder2 = DecoderBlock(2 * filters[2], filters[1])
#         self.decoder1 = DecoderBlock(2 * filters[1], filters[0])
#         self.decoder0 = DecoderBlock(2 * filters[0], filters[0])
#
#         # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
#         self.finaldeconv1 = nn.Conv2d(filters[0], 32, 3, padding=1)
#         self.finalrelu1 = nonlinearity
#         self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
#         # self.finalconv2 = nn.Conv2d(64, 32, 3, padding=1)
#         self.finalrelu2 = nonlinearity
#         self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
#
#     def forward(self, x, input2):
#         # Encoder
#         feat1 = self.resnet(x[:, 0:3, :, :])
#         e0, e1, e2, e3 = feat1[0],feat1[1],feat1[2],feat1[3]
#
#         # feat2 = self.resnet(input2)
#         # e20, e21, e22, e23, e24 = feat2[0], feat2[1], feat2[2], feat2[3], feat2[4]
#         e00,e01,e02  = self.encod(input2)  #16,32,64
#         e0 = self.c0(e0, e01)
#         e1 = self.c1(e1, e02)
#         # e1 = self.c1(e11, e21)
#         # e2 = self.c2(e12, e22)
#         # e3 = self.c3(e13, e23)
#         # e4 = self.c4(e14, e24)
#
#         # e3 = self.dblock(e3)
#
#         # Decoder
#         # e2 = self.pam2(e2)
#         # e3 = self.pam3(e3)
#         # d4 = self.decoder4(e4)
#         d3,_ = self.decoder3(e3)
#         d2,feat2 = self.decoder2(self.cam(torch.cat((d3, e2), dim=1)))
#         d1,_ = self.decoder1(self.cam(torch.cat((d2, e1), dim=1)))
#         d0,_ = self.decoder0(self.cam(torch.cat((d1, e0), dim=1)))
#
#         # e3 = self.pam3(e3)
#         # d4 = self.decoder4(e4) + e3
#         # d3 = self.decoder3(d4) + e2
#         # d2 = self.decoder2(d3) + e1
#         # d1 = self.decoder1(d2)
#         # d0 = self.cam(torch.cat((d0, e00), dim=1))
#
#         res1 = self.finaldeconv1(d0)
#         res1 = self.finalrelu1(res1)
#         out = self.finalconv2(res1)
#         out = self.finalrelu2(out)
#         out = self.finalconv3(out)
#         return out, feat2


class RegNet_MultiHeadU3_Land_Con(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(RegNet_MultiHeadU3_Land_Con, self).__init__()

        # filters = [32, 64, 128, 320, 768]
        spatial=[32, 64]
        filters = [32, 144, 288, 576]
        self.resnet = regnety_064(pretrained=True, features_only=True)
        # self.encod = UPBlock2(10,filters[0],filters[1])
        # self.encod = UPBlock2(10, spatial[0], spatial[1])
        self.encod = UPBlock2026(10, spatial[0], spatial[1])

        self.c0 = FU_Block2025(spatial[0],filters[0])
        self.c1 = FU_Block2025(spatial[1],filters[1])

        self.pam2 = PositionAttentionModule(in_channels=filters[2])
        self.pam3 = PositionAttentionModule(in_channels=filters[3])
        self.dblock = Dblock(filters[3])
        self.cam = ChannelAttentionModule()

        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(2 * filters[2], filters[1])
        self.decoder1 = DecoderBlock(2 * filters[1], filters[0])
        self.decoder0 = DecoderBlock(2 * filters[0], filters[0])

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv1 = nn.Conv2d(filters[0], 32, 3, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalconv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x, input2):
        # Encoder
        feat1 = self.resnet(x[:, 0:3, :, :])
        e0, e1, e2, e3 = feat1[0],feat1[1],feat1[2],feat1[3]

        # feat2 = self.resnet(input2)
        # e20, e21, e22, e23, e24 = feat2[0], feat2[1], feat2[2], feat2[3], feat2[4]
        e00,e01,e02  = self.encod(input2)  #16,32,64
        e0 = self.c0(e0, e01)
        e1 = self.c1(e1, e02)
        # e1 = self.c1(e11, e21)
        # e2 = self.c2(e12, e22)
        # e3 = self.c3(e13, e23)
        # e4 = self.c4(e14, e24)

        # e3 = self.dblock(e3)

        # Decoder
        # e2 = self.pam2(e2)
        # e3 = self.pam3(e3)
        # d4 = self.decoder4(e4)
        d3,_ = self.decoder3(e3)
        d2,feat2 = self.decoder2(self.cam(torch.cat((d3, e2), dim=1)))
        d1,_ = self.decoder1(self.cam(torch.cat((d2, e1), dim=1)))
        d0,_ = self.decoder0(self.cam(torch.cat((d1, e0), dim=1)))

        # e3 = self.pam3(e3)
        # d4 = self.decoder4(e4) + e3
        # d3 = self.decoder3(d4) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)
        # d0 = self.cam(torch.cat((d0, e00), dim=1))

        res1 = self.finaldeconv1(d0)
        res1 = self.finalrelu1(res1)
        out = self.finalconv2(res1)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out, feat2

class RegNet_MultiHeadU3_Land_Con_Baseline(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(RegNet_MultiHeadU3_Land_Con_Baseline, self).__init__()

        # filters = [32, 64, 128, 320, 768]
        spatial=[32, 64]
        filters = [32, 144, 288, 576]
        self.resnet = regnety_064(pretrained=True, features_only=True)
        # self.encod = UPBlock2(10,filters[0],filters[1])
        # self.encod = UPBlock2(10, spatial[0], spatial[1])
        self.encod = UPBlock2026(10, spatial[0], spatial[1])

        self.c0 = FU_Block2025(spatial[0],filters[0])
        self.c1 = FU_Block2025(spatial[1],filters[1])

        self.pam2 = PositionAttentionModule(in_channels=filters[2])
        self.pam3 = PositionAttentionModule(in_channels=filters[3])
        self.dblock = Dblock(filters[3])
        self.cam = ChannelAttentionModule()

        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(2 * filters[2], filters[1])
        self.decoder1 = DecoderBlock(2 * filters[1], filters[0])
        self.decoder0 = DecoderBlock(2 * filters[0], filters[0])

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv1 = nn.Conv2d(filters[0], 32, 3, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalconv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x, input2):
        # Encoder
        feat1 = self.resnet(x[:, 0:3, :, :])
        e0, e1, e2, e3 = feat1[0],feat1[1],feat1[2],feat1[3]

        d3,_ = self.decoder3(e3)
        d2,_ = self.decoder2(torch.cat((d3, e2), dim=1))
        d1,_ = self.decoder1(torch.cat((d2, e1), dim=1))
        d0,_ = self.decoder0(torch.cat((d1, e0), dim=1))

        res1 = self.finaldeconv1(d0)
        res1 = self.finalrelu1(res1)
        out = self.finalconv2(res1)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out

class RegNet_MultiHeadU3_Land(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(RegNet_MultiHeadU3_Land, self).__init__()

        # filters = [32, 64, 128, 320, 768]
        spatial=[32, 64]
        filters = [32, 144, 288, 576]
        self.resnet = regnety_064(pretrained=True, features_only=True)
        # self.encod = UPBlock2(10,filters[0],filters[1])
        # self.encod = UPBlock2(10, spatial[0], spatial[1])
        self.encod = UPBlock2026(10, spatial[0], spatial[1])

        self.c0 = FU_Block2025(spatial[0],filters[0])
        self.c1 = FU_Block2025(spatial[1],filters[1])

        self.pam2 = PositionAttentionModule(in_channels=filters[2])
        self.pam3 = PositionAttentionModule(in_channels=filters[3])
        self.dblock = Dblock(filters[3])
        self.cam = ChannelAttentionModule()

        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(2 * filters[2], filters[1])
        self.decoder1 = DecoderBlock(2 * filters[1], filters[0])
        self.decoder0 = DecoderBlock(2 * filters[0], filters[0])

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv1 = nn.Conv2d(filters[0], 32, 3, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalconv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x, input2):
        # Encoder
        feat1 = self.resnet(x[:, 0:3, :, :])
        e0, e1, e2, e3 = feat1[0],feat1[1],feat1[2],feat1[3]

        # feat2 = self.resnet(input2)
        # e20, e21, e22, e23, e24 = feat2[0], feat2[1], feat2[2], feat2[3], feat2[4]
        e00,e01,e02  = self.encod(input2)  #16,32,64
        e0 = self.c0(e0, e01)
        e1 = self.c1(e1, e02)
        # e1 = self.c1(e11, e21)
        # e2 = self.c2(e12, e22)
        # e3 = self.c3(e13, e23)
        # e4 = self.c4(e14, e24)

        e3 = self.dblock(e3)

        # Decoder
        # e2 = self.pam2(e2)
        # e3 = self.pam3(e3)
        # d4 = self.decoder4(e4)
        d3 = self.decoder3(e3)
        d2 = self.decoder2(self.cam(torch.cat((d3, e2), dim=1)))
        d1 = self.decoder1(self.cam(torch.cat((d2, e1), dim=1)))
        d0 = self.decoder0(self.cam(torch.cat((d1, e0), dim=1)))

        # e3 = self.pam3(e3)
        # d4 = self.decoder4(e4) + e3
        # d3 = self.decoder3(d4) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)
        # d0 = self.cam(torch.cat((d0, e00), dim=1))

        out = self.finaldeconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out


class RegNet_MultiHeadU3_2026_noS(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(RegNet_MultiHeadU3_2026_noS, self).__init__()

        # filters = [32, 64, 128, 320, 768]
        spatial=[32, 64]
        filters = [32, 144, 288, 576]
        self.resnet = regnety_064(pretrained=True, features_only=True)
        # self.encod = UPBlock2(10,filters[0],filters[1])
        self.encod = UPBlock2(10, spatial[0], spatial[1])
        # self.encod = UPBlock2026(10, spatial[0], spatial[1])

        self.c0 = FU_Block2025(spatial[0],filters[0])
        self.c1 = FU_Block2025(spatial[1],filters[1])

        self.cam = ChannelAttentionModule()

        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(2 * filters[2], filters[1])
        self.decoder1 = DecoderBlock(2 * filters[1], filters[0])
        self.decoder0 = DecoderBlock(2 * filters[0], filters[0])


        self.final_center = nn.Conv2d(filters[0], 32, kernel_size=1)
        self.final_ctr_relu1 = nonlinearity
        self.final_ctr_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_ctr_relu2 = nonlinearity
        self.final_ctr_conv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv1 = nn.Conv2d(filters[0], 32, 3, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalconv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x, input2):
        # Encoder
        feat1 = self.resnet(x[:, 0:3, :, :])
        e0, e1, e2, e3 = feat1[0],feat1[1],feat1[2],feat1[3]

        # feat2 = self.resnet(input2)
        # e20, e21, e22, e23, e24 = feat2[0], feat2[1], feat2[2], feat2[3], feat2[4]
        # e00,e01,e02  = self.encod(input2)  #16,32,64
        # e0 = self.c0(e0, e01)
        # e1 = self.c1(e1, e02)
        # e1 = self.c1(e11, e21)
        # e2 = self.c2(e12, e22)
        # e3 = self.c3(e13, e23)
        # e4 = self.c4(e14, e24)

        # e4 = self.dblock(e4)

        # Decoder
        # e3 = self.pam3(e3)
        # e4 = self.pam4(e4)
        # d4 = self.decoder4(e4)
        d3 = self.decoder3(e3)
        d2 = self.decoder2(self.cam(torch.cat((d3, e2), dim=1)))
        d1 = self.decoder1(self.cam(torch.cat((d2, e1), dim=1)))
        d0 = self.decoder0(self.cam(torch.cat((d1, e0), dim=1)))

        # e3 = self.pam3(e3)
        # d4 = self.decoder4(e4) + e3
        # d3 = self.decoder3(d4) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)
        # d0 = self.cam(torch.cat((d0, e00), dim=1))

        # # only ctrline
        # ctr = self.final_center(d0)
        # ctr = self.final_ctr_relu1(ctr)
        # ctr = self.final_ctr_conv2(ctr)
        # ctr = self.final_ctr_relu2(ctr)
        # ctr = self.final_ctr_conv3(ctr)

        out = self.finaldeconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out


class RegNet_MultiHeadU3_2026(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(RegNet_MultiHeadU3_2026, self).__init__()

        # filters = [32, 64, 128, 320, 768]
        spatial=[32, 64]
        filters = [32, 144, 288, 576]
        self.resnet = regnety_064(pretrained=True, features_only=True)
        # self.encod = UPBlock2(10,filters[0],filters[1])
        self.encod = UPBlock2(10, spatial[0], spatial[1])

        self.c0 = FU_Block2025(spatial[0],filters[0])
        self.c1 = FU_Block2025(spatial[1],filters[1])

        self.cam = ChannelAttentionModule()

        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(2 * filters[2], filters[1])
        self.decoder1 = DecoderBlock(2 * filters[1], filters[0])
        self.decoder0 = DecoderBlock(2 * filters[0], filters[0])

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv1 = nn.Conv2d(filters[0]+filters[0]//2, 32, 3, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x, input2):
        # Encoder
        feat1 = self.resnet(x[:, 0:3, :, :])
        e0, e1, e2, e3 = feat1[0],feat1[1],feat1[2],feat1[3]

        # feat2 = self.resnet(input2)
        # e20, e21, e22, e23, e24 = feat2[0], feat2[1], feat2[2], feat2[3], feat2[4]
        e00,e01,e02  = self.encod(input2)  #16,32,64
        e0 = self.c0(e0, e01)
        e1 = self.c1(e1, e02)
        # e1 = self.c1(e11, e21)
        # e2 = self.c2(e12, e22)
        # e3 = self.c3(e13, e23)
        # e4 = self.c4(e14, e24)

        # e4 = self.dblock(e4)

        # Decoder
        # e3 = self.pam3(e3)
        # e4 = self.pam4(e4)
        # d4 = self.decoder4(e4)
        d3 = self.decoder3(e3)
        d2 = self.decoder2(self.cam(torch.cat((d3, e2), dim=1)))
        d1 = self.decoder1(self.cam(torch.cat((d2, e1), dim=1)))
        d0 = self.decoder0(self.cam(torch.cat((d1, e0), dim=1)))

        # e3 = self.pam3(e3)
        # d4 = self.decoder4(e4) + e3
        # d3 = self.decoder3(d4) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)
        d0 = self.cam(torch.cat((d0, e00), dim=1))
        out = self.finaldeconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out


class RegNet_MultiHeadU3_20250113(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(RegNet_MultiHeadU3_20250113, self).__init__()

        # filters = [32, 64, 128, 320, 768]
        spatial=[32, 64]
        filters = [32, 144, 288, 576]
        self.resnet = regnety_064(pretrained=True, features_only=True)
        # self.encod = UPBlock2(10,filters[0],filters[1])
        self.encod = UPBlock2(10, spatial[0], spatial[1])

        self.c0 = FU_Block2025(spatial[0],filters[0])
        self.c1 = FU_Block2025(spatial[1],filters[1])

        # self.cam = ChannelAttentionModule()
        self.cam0 = MSCAAttention(2 * filters[2])
        self.cam1 = MSCAAttention(2 * filters[1])
        self.cam2 = MSCAAttention(2 * filters[0])
        self.cam3 = MSCAAttention(filters[0]+filters[0]//2)

        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(2 * filters[2], filters[1])
        self.decoder1 = DecoderBlock(2 * filters[1], filters[0])
        self.decoder0 = DecoderBlock(2 * filters[0], filters[0])

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv1 = nn.Conv2d(filters[0]+filters[0]//2, 32, 3, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x, input2):
        # Encoder
        feat1 = self.resnet(x[:, 0:3, :, :])
        e0, e1, e2, e3 = feat1[0],feat1[1],feat1[2],feat1[3]

        # feat2 = self.resnet(input2)
        # e20, e21, e22, e23, e24 = feat2[0], feat2[1], feat2[2], feat2[3], feat2[4]
        e00,e01,e02  = self.encod(input2)  #16,32,64
        e0 = self.c0(e0, e01)
        e1 = self.c1(e1, e02)
        # e1 = self.c1(e11, e21)
        # e2 = self.c2(e12, e22)
        # e3 = self.c3(e13, e23)
        # e4 = self.c4(e14, e24)

        # e4 = self.dblock(e4)

        # Decoder
        # e3 = self.pam3(e3)
        # e4 = self.pam4(e4)
        # d4 = self.decoder4(e4)
        d3 = self.decoder3(e3)
        d2 = self.decoder2(self.cam0(torch.cat((d3, e2), dim=1)))
        d1 = self.decoder1(self.cam1(torch.cat((d2, e1), dim=1)))
        d0 = self.decoder0(self.cam2(torch.cat((d1, e0), dim=1)))

        # e3 = self.pam3(e3)
        # d4 = self.decoder4(e4) + e3
        # d3 = self.decoder3(d4) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)
        d0 = self.cam3(torch.cat((d0, e00), dim=1))
        out = self.finaldeconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out


class HRet_MultiHeadU3_64(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(HRet_MultiHeadU3_64, self).__init__()

        # filters = [32, 64, 128, 320, 768]
        # filters = [32, 144, 288, 576]
        filters = [64, 128, 256, 512]
        self.resnet = hrnet_w32(pretrained=True, features_only=True)
        self.encod = UPBlock2(10,filters[0],filters[1])

        self.c0 = FU_Block(filters[0])
        self.c1 = FU_Block(filters[1])

        self.cam = ChannelAttentionModule()

        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(2 * filters[2], filters[1])
        self.decoder1 = DecoderBlock(2 * filters[1], filters[0])
        self.decoder0 = DecoderBlock(2 * filters[0], filters[0])

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv1 = nn.Conv2d(filters[0]+filters[0]//2, 32, 3, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x, input2):
        # Encoder
        feat1 = self.resnet(x[:, 0:3, :, :])
        e0, e1, e2, e3 = feat1[0],feat1[1],feat1[2],feat1[3]

        # feat2 = self.resnet(input2)
        # e20, e21, e22, e23, e24 = feat2[0], feat2[1], feat2[2], feat2[3], feat2[4]
        e00,e01,e02  = self.encod(input2)
        e0 = self.c0(e0, e01)
        e1 = self.c1(e1, e02)

        d3 = self.decoder3(e3)
        d2 = self.decoder2(self.cam(torch.cat((d3, e2), dim=1)))
        d1 = self.decoder1(self.cam(torch.cat((d2, e1), dim=1)))
        d0 = self.decoder0(self.cam(torch.cat((d1, e0), dim=1)))

        d0 = self.cam(torch.cat((d0, e00), dim=1))
        out = self.finaldeconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out


class RegNet_MultiHeadU3_better(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(RegNet_MultiHeadU3_better, self).__init__()

        # filters = [32, 64, 128, 320, 768]
        filters = [32, 144, 288, 576]
        self.resnet = regnety_064(pretrained=True, features_only=True)
        self.encod = UPBlockatt(10,filters[0],filters[1])

        self.c0 = FU_Block(filters[0])
        self.c1 = FU_Block(filters[1])

        self.cam = ChannelAttentionModule()

        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(2 * filters[2], filters[1])
        self.decoder1 = DecoderBlock(2 * filters[1], filters[0])
        self.decoder0 = DecoderBlock(2 * filters[0], filters[0])

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv1 = nn.Conv2d(filters[0]+filters[0]//2, 32, 3, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x, input2):
        # Encoder
        feat1 = self.resnet(x[:, 0:3, :, :])
        e0, e1, e2, e3 = feat1[0],feat1[1],feat1[2],feat1[3]

        # feat2 = self.resnet(input2)
        # e20, e21, e22, e23, e24 = feat2[0], feat2[1], feat2[2], feat2[3], feat2[4]
        e00,e01,e02  = self.encod(input2)
        e0 = self.c0(e0, e01)
        e1 = self.c1(e1, e02)
        # e1 = self.c1(e11, e21)
        # e2 = self.c2(e12, e22)
        # e3 = self.c3(e13, e23)
        # e4 = self.c4(e14, e24)

        # e4 = self.dblock(e4)

        # Decoder
        # e3 = self.pam3(e3)
        # e4 = self.pam4(e4)
        # d4 = self.decoder4(e4)
        d3 = self.decoder3(e3)
        d2 = self.decoder2(self.cam(torch.cat((d3, e2), dim=1)))
        d1 = self.decoder1(self.cam(torch.cat((d2, e1), dim=1)))
        d0 = self.decoder0(self.cam(torch.cat((d1, e0), dim=1)))

        # e3 = self.pam3(e3)
        # d4 = self.decoder4(e4) + e3
        # d3 = self.decoder3(d4) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)
        d0 = self.cam(torch.cat((d0, e00), dim=1))
        out = self.finaldeconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out


class RegNet_MultiEncode(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(RegNet_MultiEncode, self).__init__()

        filters = [32, 64, 128, 320, 768]
        self.resnet = regnety_008(pretrained=True, features_only=True)
        self.encod = ECBlock(3,filters[0])

        self.cam = ChannelAttentionModule()

        self.c2 = FUSION_Block(filters[2])
        self.c1 = FUSION_Block(filters[1])
        self.c0 = FUSION_Block(filters[0])
        # self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(filters[2], filters[1])
        self.decoder1 = DecoderBlock(2 * filters[1], filters[0])
        self.decoder0 = DecoderBlock(2 * filters[0], filters[0])

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldedconv1 = nn.Conv2d(filters[0], 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nonlinearity
        self.finaldconv = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x, input2):
        # Encoder
        feat1 = self.resnet(x[:, 0:3, :, :])
        e10, e11, e12= feat1[0],feat1[1],feat1[2]
        feat2 = self.resnet(input2)
        e20, e21, e22= feat2[0], feat2[1], feat2[2]

        e0 = self.c0(e10, e20)
        e1 = self.c1(e11, e21)
        e2 = self.c2(e12, e22)

        d2 = self.decoder2(e2)
        d1 = self.decoder1(self.cam(torch.cat((d2+e1, e1), dim=1)))
        d0 = self.decoder0(self.cam(torch.cat((d1+e0, e0), dim=1)))

        # d0 = self.cam(torch.cat((d0, e00), dim=1))
        out = self.finaldedconv1(d0)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.finaldconv(out)
        return out


class Net_TesHR(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(Net_TesHR, self).__init__()

        # filters = [64, 128, 256, 512]
        # resnet = models.resnet34(pretrained=True)
        addfilt=32
        filters = [64, 128, 320, 768]
        self.resnet = hrnet_w18(pretrained=True, features_only=True)

        self.c4 = DF_Block6(768)
        self.c3 = DF_Block6(320)
        self.c2 = DF_Block6(128)
        self.c1 = DF_Block6(64)
        self.c0 = DF_Block6(32)

        # filters = [144, 288, 576, 1296]
        # self.resnet = regnety_064(pretrained=True, features_only=True)
        # filters = [48,96,160,304]
        # self.resnet = edgenext_small(pretrained=True, features_only=True)
        # self.resnet = regnetz_040(pretrained=True, features_only=True)
        # filters = [96, 192, 384, 768]
        # self.resnet = convnext_tiny(pretrained=True, features_only=True)
        # self.resnet = resnest26d(pretrained=True, features_only=True)

        # self.dblock = Dblock(768)

        # self.pam1 = PositionAttentionModule(in_channels=64)
        # self.pam2 = PositionAttentionModule(in_channels=128)
        self.pam3 = PositionAttentionModule(in_channels=filters[2])
        self.pam4 = PositionAttentionModule(in_channels=filters[3])
        self.cam = ChannelAttentionModule()
        # self.decoder4 = DecoderBlock(filters[3], filters[2])
        # self.decoder3 = DecoderBlock(filters[2], filters[1])
        # self.decoder2 = DecoderBlock(filters[1], filters[0])
        # self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(2 * filters[2], filters[1])
        self.decoder2 = DecoderBlock(2 * filters[1], filters[0])
        self.decoder1 = DecoderBlock(2 * filters[0], addfilt)
        self.decoder0 = DecoderBlock(2*addfilt, addfilt)

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv1 = nn.Conv2d(addfilt, 32, 3, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x, input2):
        # Encoder
        feat1 = self.resnet(x[:, 0:3, :, :])
        e10, e11, e12, e13, e14 = feat1[0],feat1[1],feat1[2],feat1[3],feat1[4]

        feat2 = self.resnet(input2)
        e20, e21, e22, e23, e24 = feat2[0], feat2[1], feat2[2], feat2[3], feat2[4]

        e0 = self.c0(e10, e20)
        e1 = self.c1(e11, e21)
        e2 = self.c2(e12, e22)
        e3 = self.c3(e13, e23)
        e4 = self.c4(e14, e24)

        # e4 = self.dblock(e4)

        # Decoder
        e3 = self.pam3(e3)
        e4 = self.pam4(e4)
        d4 = self.decoder4(e4)
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))
        d0 = self.decoder0(torch.cat((d1, e0), dim=1))

        # e3 = self.pam3(e3)
        # d4 = self.decoder4(e4) + e3
        # d3 = self.decoder3(d4) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)
        d0 = self.cam(d0)
        out = self.finaldeconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out


class AttRegnet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(AttRegnet34, self).__init__()

        # filters = [64, 128, 256, 512]
        # resnet = models.resnet34(pretrained=True)

        filters = [64, 128, 320, 768]
        self.resnet = regnety_008(pretrained=True, features_only=True)

        # self.dblock = Dblock(768)

        # self.pam1 = PositionAttentionModule(in_channels=64)
        # self.pam2 = PositionAttentionModule(in_channels=128)
        self.pam3 = PositionAttentionModule(in_channels=320)
        self.cam = ChannelAttentionModule()
        # self.decoder4 = DecoderBlock(filters[3], filters[2])
        # self.decoder3 = DecoderBlock(filters[2], filters[1])
        # self.decoder2 = DecoderBlock(filters[1], filters[0])
        # self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(2 * filters[2], filters[1])
        self.decoder2 = DecoderBlock(2 * filters[1], filters[0])
        self.decoder1 = DecoderBlock(2 * filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        feat = self.resnet(x[:, 0:3, :, :])
        e1, e2, e3, e4 = feat[1],feat[2],feat[3],feat[4]
        # e1, e2, e3, e4 = self.resnet(x[:, 0:3, :, :])
        # Center
        e4 = self.cam(e4)
        # e4 = self.dblock(e4)

        # Decoder
        e3 = self.pam3(e3)
        d4 = self.decoder4(e4)
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))

        # e3 = self.pam3(e3)
        # d4 = self.decoder4(e4) + e3
        # d3 = self.decoder3(d4) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out

class DecodBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecodBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels // 2)
        self.deconv2 = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 2)
        self.conv3 = nn.Conv2d(in_channels // 2, out_channels, 3, padding=1)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class RegNet_MHead(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(RegNet_MHead, self).__init__()

        filters = [32, 64, 128, 320, 768]
        self.resnet = regnety_008(pretrained=True, features_only=True)

        self.encod = ECBlock(3,filters[0])
        self.c0 = FU_Block(filters[0])

        self.cam = ChannelAttentionModule()

        self.decoder3 = DecodBlock(filters[3], filters[2])
        self.decoder2 = DecodBlock(2 * filters[2], filters[1])
        self.decoder1 = DecodBlock(2 * filters[1], filters[0])
        self.decoder0 = DecodBlock(2 * filters[0], filters[0])

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv1 = nn.Conv2d(filters[0]+16, 32, 3, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x, input2):
        # Encoder
        feat1 = self.resnet(x[:, 0:3, :, :])
        e0, e1, e2, e3 = feat1[0],feat1[1],feat1[2],feat1[3]

        # feat2 = self.resnet(input2)
        # e20, e21, e22, e23, e24 = feat2[0], feat2[1], feat2[2], feat2[3], feat2[4]
        e00,e01  = self.encod(input2)
        e0 = self.c0(e0, e01)
        # e1 = self.c1(e11, e21)
        # e2 = self.c2(e12, e22)
        # e3 = self.c3(e13, e23)
        # e4 = self.c4(e14, e24)

        # e4 = self.dblock(e4)

        # Decoder
        # e3 = self.pam3(e3)
        # e4 = self.pam4(e4)
        # d4 = self.decoder4(e4)
        d3 = self.decoder3(e3)
        d2 = self.decoder2(self.cam(torch.cat((d3, e2), dim=1)))
        d1 = self.decoder1(self.cam(torch.cat((d2, e1), dim=1)))
        d0 = self.decoder0(self.cam(torch.cat((d1, e0), dim=1)))

        # e3 = self.pam3(e3)
        # d4 = self.decoder4(e4) + e3
        # d3 = self.decoder3(d4) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)
        d0 = self.cam(torch.cat((d0, e00), dim=1))
        out = self.finaldeconv1(d0)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out



class AttHRNet32(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(AttHRNet32, self).__init__()

        filters = [128, 256, 512, 1024]
        self.resnet = hrnet_w32(pretrained=True, features_only=True)
        # resnet = models.resnet34(pretrained=True)

        # filters = [64, 128, 320, 768]
        # self.resnet = regnety_008(pretrained=True)

        # self.dblock = Dblock(768)

        # self.pam1 = PositionAttentionModule(in_channels=64)
        self.pam3 = PositionAttentionModule(in_channels=filters[2])
        self.pam4 = PositionAttentionModule(in_channels=filters[3])
        self.cam = ChannelAttentionModule()
        # self.decoder4 = DecoderBlock(filters[3], filters[2])
        # self.decoder3 = DecoderBlock(filters[2], filters[1])
        # self.decoder2 = DecoderBlock(filters[1], filters[0])
        # self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(2 * filters[2], filters[1])
        self.decoder2 = DecoderBlock(2 * filters[1], filters[0])
        self.decoder1 = DecoderBlock(2 * filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        # e1, e2, e3, e4 = self.resnet(x[:, 0:3, :, :])
        feat = self.resnet(x[:, 0:3, :, :])
        e1, e2, e3, e4 = feat[1],feat[2],feat[3],feat[4]
        # Center
        e3 = self.pam3(e3)
        e4 = self.pam4(e4)
        e4 = self.cam(e4)
        # e4 = self.dblock(e4)
        # Decoder

        d4 = self.decoder4(e4)
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))

        # e3 = self.pam3(e3)
        # d4 = self.decoder4(e4) + e3
        # d3 = self.decoder3(d4) + e2
        # d2 = self.decoder2(d3) + e1
        # d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out

class DinkNet50(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet50, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x[:,0:3,:,:])
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out


class DinkNet101(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet101, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet101(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class LinkNet34(nn.Module):
    def __init__(self, num_classes=1):
        super(LinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)