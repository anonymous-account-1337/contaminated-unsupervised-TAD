import torch
import torch.nn as nn


def conv_bn_relu(in_c, out_c, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )


class InceptionBlockV3(nn.Module):

    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, pool_proj, dropout=0.0):
        super().__init__()
        self.branch1 = conv_bn_relu(in_channels, out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_bn_relu(in_channels, red_5x5, kernel_size=1),
            conv_bn_relu(red_5x5, out_5x5, kernel_size=3, padding=1),
            conv_bn_relu(out_5x5, out_5x5, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            conv_bn_relu(in_channels, red_3x3, kernel_size=1),
            conv_bn_relu(red_3x3, out_3x3, kernel_size=(1, 3), padding=(0, 1)),
            conv_bn_relu(out_3x3, out_3x3, kernel_size=(3, 1), padding=(1, 0)),
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            conv_bn_relu(in_channels, pool_proj, kernel_size=1)
        )

        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        squeeze_batch = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_batch = True
        elif x.dim() != 4:
            raise ValueError('Input must be (C,H,W) or (N,C,H,W)')

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.dropout(out)

        if squeeze_batch:
            out = out.squeeze(0)

        return out


class InceptionBlockV1(nn.Module):

    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, pool_proj, dropout=0.0):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        squeeze_batch = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_batch = True
        elif x.dim() != 4:
            raise ValueError('Input must be (C,H,W) or (N,C,H,W)')

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.dropout(out)

        if squeeze_batch:
            out = out.squeeze(0)

        return out


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels // 2 * 2 + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
