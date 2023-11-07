import torch
from torch import nn
from torchvision import models

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        for rate in atrous_rates:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        self.convs = nn.ModuleList(modules)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv1x1 = nn.Conv2d(len(atrous_rates) * out_channels + out_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[2:]
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res.append(F.interpolate(self.global_avg_pool(x), size=size, mode='bilinear', align_corners=True))
        res = torch.cat(res, dim=1)
        res = self.conv1x1(res)
        res = self.bn(res)
        return self.relu(res)

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(DeepLabHead, self).__init__()
        self.aspp = ASPP(in_channels, 256, atrous_rates)
        self.conv = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.final_conv = nn.Conv2d(256, out_channels, 1)

    def forward(self, x):
        x = self.aspp(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.final_conv(x)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter

class DeepLabV3(nn.Module):
    def __init__(self, num_classes, backbone_name='resnet101', pretrained_backbone=True, atrous_rates=(12, 24, 36)):
        super(DeepLabV3, self).__init__()
        if backbone_name == 'resnet101':
            backbone = models.resnet101(pretrained=pretrained_backbone)
            # Specify the return layers as desired
            return_layers = {'layer4': 'out', 'layer1': 'low_level'}
            self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
            low_level_channels = 256  # The number of channels after layer1 in ResNet101
            high_level_channels = 2048  # The number of channels after layer4 in ResNet101
        else:
            raise NotImplementedError('Only Resnet101 backbone is implemented')

        self.head = DeepLabHead(high_level_channels, num_classes, atrous_rates)
        self.low_level_conv = nn.Conv2d(low_level_channels, 99, 1)
        self.final_conv = nn.Conv2d(100, num_classes, 1)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        low_level_feat = features['low_level']
        high_level_feat = features['out']

        # Reduce the number of channels for the low-level features
        low_level_feat = self.low_level_conv(low_level_feat)
        print(low_level_feat.shape)

        # Process the high-level features through the head
        high_level_feat = self.head(high_level_feat)
    

        # Before concatenating, ensure that the high-level features are upsampled to the same size as the low-level features
        high_level_feat_upsampled = F.interpolate(high_level_feat, size=low_level_feat.shape[-2:], mode='bilinear', align_corners=False)

        # Before the final convolution, ensure the concatenated features have the expected number of channels
        concat_features = torch.cat((low_level_feat, high_level_feat_upsampled), dim=1)

        
    

        # Final convolution
        output = self.final_conv(concat_features)

        # Upsample to the input size
        output = F.interpolate(output, size=input_shape, mode='bilinear', align_corners=False)

        return output




