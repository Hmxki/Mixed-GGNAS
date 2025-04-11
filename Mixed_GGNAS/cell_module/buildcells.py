import torch.nn as nn
from .cells.raw_cells import *
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super(SpatialAttention, self).__init__()
        self.squeeze = nn.Conv2d(dim, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super(ChannelAttention, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(dim, dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(dim // reduction, dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z

class buildcell(nn.Module):
    def __init__(self, id,in_channels,out_channels,cell_num, weight,retrain=False,new_w=None):
        super(buildcell, self).__init__()

        self.id = id
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cell_num = cell_num
        self.weight =weight
        self.cell_type='conv'
        self.retrain = retrain
        self.new_w = new_w


        if id<4:
            #编码单元
            self.pool = nn.Sequential(
                            nn.BatchNorm2d(self.in_channels),
                            nn.AvgPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(self.in_channels,self.out_channels,3,padding=1,groups=16)
                            )

        else:
            # 解码单元
            self.bm = nn.BatchNorm2d(self.in_channels)
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

            self.bm1 = nn.BatchNorm2d(self.in_channels)
            self.up1 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

            self.satt = SpatialAttention(self.in_channels//2)
            self.catt = ChannelAttention(self.in_channels//2, reduction=4)
            self.conv1x1 = nn.Conv2d(self.in_channels, self.in_channels//2, kernel_size=1)

        #self.adj_channel = nn.Conv2d(self.in_channels, self.out_channels, 1)


        if self.cell_num == 0:
            # resblock

            self.cell_0 = ResidualBlock(in_channels=self.out_channels,out_channels=self.out_channels,
                                        kernel_size=3)
            self.cell_1 = ResidualBlock(in_channels=self.out_channels, out_channels=self.out_channels,
                                        kernel_size=5)
            self.cell_2 = ResidualBlock(in_channels=self.out_channels, out_channels=self.out_channels,
                                        kernel_size=7)

        elif self.cell_num == 1:
            # denseblock
            self.cell_0 = DenseBlock(in_channels=self.out_channels, out_channels=self.out_channels,
                                        kernel_size=3)
            self.cell_1 = DenseBlock(in_channels=self.out_channels, out_channels=self.out_channels,
                                        kernel_size=5)
            self.cell_2 = DenseBlock(in_channels=self.out_channels, out_channels=self.out_channels,
                                        kernel_size=7)

        elif self.cell_num == 2:
            # inceptionblock
            self.cell_0 = InceptionBlock(in_channels=self.out_channels, out_channels=self.out_channels,
                                     kernel_size=3)
            self.cell_1 = InceptionBlock(in_channels=self.out_channels, out_channels=self.out_channels,
                                     kernel_size=5)
            self.cell_2 = InceptionBlock(in_channels=self.out_channels, out_channels=self.out_channels,
                                     kernel_size=7)

        else:
            # convnextblock
            self.cell_0 = ConvNextBlock(in_channels=self.out_channels, out_channels=self.out_channels,
                                         kernel_size=3)
            self.cell_1 = ConvNextBlock(in_channels=self.out_channels, out_channels=self.out_channels,
                                         kernel_size=5)
            self.cell_2 = ConvNextBlock(in_channels=self.out_channels, out_channels=self.out_channels,
                                         kernel_size=7)



    def forward(self, inputs,en_out=None):
        if self.id<4:
            dw_up = self.pool(inputs)
        else:
            bm = self.bm(inputs)
            dw_up = self.up(bm)
            # en_out = en_out[:, :en_out.size()[1] // 2, :, :]
            en_out = self.bm1(en_out)
            en_out = self.up1(en_out)

            diff_y = en_out.size()[2] - dw_up.size()[2]
            diff_x = en_out.size()[3] - dw_up.size()[3]

            # padding_left, padding_right, padding_top, padding_bottom
            dw_up = F.pad(dw_up, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])

            satt = self.satt(en_out)
            catt = self.catt(en_out)
            en_out = en_out + (satt * catt)
            dw_up = self.conv1x1(torch.cat([en_out, dw_up], dim=1))

        #dw_up_adj_channel = self.adj_channel(dw_up)
        if self.retrain==False:
            cell_0 = self.cell_0(dw_up)
            cell_1 = self.cell_1(dw_up)
            cell_2 = self.cell_2(dw_up)
            cell_3 = cell_0+dw_up
            cell_4 = cell_1 + dw_up
            cell_5 = cell_2 + dw_up
            self.weight.to('cuda:0')
            out = cell_0*self.weight[0]+cell_1*self.weight[1]+cell_2*self.weight[2]+cell_3*self.weight[3]+cell_4*self.weight[4]+cell_5*self.weight[5]
        else:
            # 重新训练，判断是哪一条路径
            index = self.new_w[self.id]
            if index==0:
                out = self.cell_0(dw_up)
            elif index==1:
                out = self.cell_1(dw_up)
            elif index==2:
                out = self.cell_2(dw_up)
            elif index==3:
                out = self.cell_0(dw_up)+dw_up
            elif index==4:
                out = self.cell_1(dw_up)+dw_up
            elif index==5:
                out = self.cell_2(dw_up)+dw_up

        return out
