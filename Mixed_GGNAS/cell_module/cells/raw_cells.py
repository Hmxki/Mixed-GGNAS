import torch
import torch.nn as nn
from Mixed_GGNAS.cell_module.cells.build_darts_cell import Build_Darts_Cell


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        if kernel_size==3:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                                   padding=1, bias=False,groups=16)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                                   padding=1, bias=False,groups=16)
        elif kernel_size==5:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                                   padding=3, dilation=3,bias=False,groups=16)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                                   padding=3, dilation=3, bias=False,groups=16)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1,
                                   padding=6, dilation=3, bias=False, groups=16)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1,
                                   padding=6,dilation=3, bias=False,groups=16)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)
        # 添加一个卷积层来调整输入张量的通道数
        if in_channels != out_channels:
            self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        else:
            self.conv_res = None
        self.relu3 = nn.ReLU(inplace=False)


    def forward(self, x):
        #print('ResidualBlock')
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        # 如果输入张量的通道数与输出张量不一致，则使用卷积层调整输入张量的通道数
        if self.conv_res:
            residual = self.conv_res(residual)

        out = residual+out
        out = self.relu3(out)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate=32, num_layers=3, kernel_size=3):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_dense_layer(in_channels + i * growth_rate, growth_rate, kernel_size))

        self.adjusts = nn.Conv2d(in_channels+num_layers*growth_rate,out_channels,1)

    def _make_dense_layer(self, in_channels, out_channels_,kernel_size):
        if kernel_size==3:
            return nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels, out_channels_, kernel_size=3, stride=1, padding=1,
                          bias=False,groups=16)
            )
        elif kernel_size==5:
            return nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels, out_channels_, kernel_size=3, stride=1, padding=3,
                          bias=False,dilation=3,groups=16)
            )
        else:
            return nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels, out_channels_, kernel_size=5, stride=1, padding=6,
                          bias=False,dilation=3,groups=16)
            )





    def forward(self, x):
        #print('DenseBlock')
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        cat = torch.cat(features,  dim=1)
        out = self.adjusts(cat)
        return out


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(InceptionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,3,padding=1,groups=16)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.relu0 = nn.ReLU(inplace=False)
        if kernel_size==3:
            self.conv0_1 = nn.Conv2d(in_channels, in_channels, (1, 3), padding=(0, 1), groups=16)
            self.conv0_2 = nn.Conv2d(in_channels, in_channels, (3, 1), padding=(1, 0), groups=16)
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.relu1 = nn.ReLU(inplace=False)

            self.conv1_1 = nn.Conv2d(in_channels, in_channels, (1, 5), padding=(0, 2), groups=16)
            self.conv1_2 = nn.Conv2d(in_channels, in_channels, (5, 1), padding=(2, 0), groups=16)
            self.bn2 = nn.BatchNorm2d(in_channels)
            self.relu2 = nn.ReLU(inplace=False)

            self.conv2_1 = nn.Conv2d(in_channels, in_channels, (1, 7), padding=(0, 3), groups=16)
            self.conv2_2 = nn.Conv2d(in_channels, in_channels, (7, 1), padding=(3, 0), groups=16)
            self.bn3 = nn.BatchNorm2d(in_channels)
            self.relu3 = nn.ReLU(inplace=False)
        elif kernel_size==5:
            self.conv0_1 = nn.Conv2d(in_channels, in_channels, (1, 5), padding=(0, 2), groups=16)
            self.conv0_2 = nn.Conv2d(in_channels, in_channels, (5, 1), padding=(2, 0), groups=16)
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.relu1 = nn.ReLU(inplace=False)

            self.conv1_1 = nn.Conv2d(in_channels, in_channels, (1, 7), padding=(0, 3), groups=16)
            self.conv1_2 = nn.Conv2d(in_channels, in_channels, (7, 1), padding=(3, 0), groups=16)
            self.bn2 = nn.BatchNorm2d(in_channels)
            self.relu2 = nn.ReLU(inplace=False)

            self.conv2_1 = nn.Conv2d(in_channels, in_channels, (1, 9), padding=(0, 4), groups=16)
            self.conv2_2 = nn.Conv2d(in_channels, in_channels, (9, 1), padding=(4, 0), groups=16)
            self.bn3 = nn.BatchNorm2d(in_channels)
            self.relu3 = nn.ReLU(inplace=False)
        else:
            self.conv0_1 = nn.Conv2d(in_channels, in_channels, (1, 3), padding=(0, 1), groups=16)
            self.conv0_2 = nn.Conv2d(in_channels, in_channels, (3, 1), padding=(1, 0), groups=16)
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.relu1 = nn.ReLU(inplace=False)

            self.conv1_1 = nn.Conv2d(in_channels, in_channels, (1, 7), padding=(0, 3), groups=16)
            self.conv1_2 = nn.Conv2d(in_channels, in_channels, (7, 1), padding=(3, 0), groups=16)
            self.bn2 = nn.BatchNorm2d(in_channels)
            self.relu2 = nn.ReLU(inplace=False)

            self.conv2_1 = nn.Conv2d(in_channels, in_channels, (1, 11), padding=(0, 5), groups=16)
            self.conv2_2 = nn.Conv2d(in_channels, in_channels, (11, 1), padding=(5, 0), groups=16)
            self.bn3 = nn.BatchNorm2d(in_channels)
            self.relu3 = nn.ReLU(inplace=False)


        self.adjust = nn.Conv2d(in_channels*3,out_channels,1)

    def forward(self, x):
        #print('InceptionBlock')
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu0(x)
        branch1_0 = self.conv0_1(x)
        branch1_1 = self.conv0_2(branch1_0)
        branch1 = self.bn1(branch1_1)
        branch1 = self.relu1(branch1)

        branch2_0 = self.conv1_1(x)
        branch2_1 = self.conv1_2(branch2_0)
        branch2 = self.bn2(branch2_1)
        branch2 = self.relu2(branch2)

        branch3_0 = self.conv2_1(x)
        branch3_1 = self.conv2_2(branch3_0)
        branch3 = self.bn3(branch3_1)
        branch3 = self.relu3(branch3)

        cat = torch.cat((branch1,branch2,branch3),dim=1)

        out = self.adjust(cat)
        return out

class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(ConvNextBlock, self).__init__()
        if kernel_size==3:
            ks=5
        elif kernel_size==5:
            ks=7
        else:
            ks=9
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=ks, groups=in_channels, padding=(ks - 1) // 2)  # depthwise conv
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.pwconv1 = nn.Linear(in_channels, 4 * in_channels)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()

        self.pwconv2 = nn.Linear(4 * in_channels, in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.GELU()

        self.adjust = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        #print('ConvNextBlock')
        residual = x
        x = self.dwconv(x)
        x = self.bn0(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)

        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.bn1(x)
        out = self.adjust(self.act1(residual + x))
        return out






# if __name__=="__main__":
#     # 随机生成输入张量数据
#     gatt = True
#     batch_size = 1
#     in_channels = 32
#     out_channels = 64
#     height, width = 32, 32
#     skip_connecttion = True
#     x = torch.randn(batch_size, in_channels, height, width)

    # 测试 ResidualBlock
    # kernel_size = 7
    # residual_block = ResidualBlock(in_channels, out_channels, kernel_size=kernel_size)
    # out = residual_block(x)
    # print("ResidualBlock 输出尺寸:", out.shape)

    # # 测试 DenseBlock
    # growth_rate = 32
    # num_layers = 3
    # out_channels = 64
    # kernel_size = 7
    # dense_block = DenseBlock(in_channels, out_channels, growth_rate, num_layers, kernel_size=kernel_size)
    # out = dense_block(x)
    # print("DenseBlock 输出尺寸:", out.shape)

    # # 测试 InceptionBlock
    # kernel_size = 5
    # inception_block = InceptionBlock(in_channels, out_channels, kernel_size=kernel_size)
    # out = inception_block(x)
    # print("InceptionBlock 输出尺寸:", out.shape)
    #
    # kernel_size = 9
    # convnextblock = ConvNextBlock(in_channels, out_channels, kernel_size, gatt=gatt, skip_connecttion=skip_connecttion)
    # out = convnextblock(x)
    # print("ConvNextBlock 输出尺寸:", out.shape)
