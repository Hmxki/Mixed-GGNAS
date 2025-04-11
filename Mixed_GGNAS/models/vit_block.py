import math

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn.modules.utils import _pair
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm



class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if upsample else nn.Identity()
        self.conv = Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class ViTDecoder(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super(ViTDecoder, self).__init__()
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(hidden_size, hidden_size//2),  # 第一个上采样块
            DecoderBlock(hidden_size//2, hidden_size//4),  # 第二个上采样块
            DecoderBlock(hidden_size//4, hidden_size//8),  # 第三个上采样块
            DecoderBlock(hidden_size//8, hidden_size//16)  # 输出通道数应与原始图像匹配，不进行额外的上采样
        ])
        self.stem = nn.Conv2d(hidden_size//16,out_channels,1)

    def forward(self, x):
        for block in self.decoder_blocks:
            x = block(x)
        return self.stem(x)



def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Mlp(nn.Module):
    def __init__(self, hidden_size,mlp_dim,dropout_rate):
        super(Mlp, self).__init__()
        self.fc1 = Linear(hidden_size, mlp_dim)
        self.fc2 = Linear(mlp_dim, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Attention(nn.Module):
    def __init__(self, num_heads, hidden_size, attention_dropout_rate, vis):
        super(Attention, self).__init__()
        self.vis = vis
        #
        self.num_attention_heads = num_heads
        #
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        #
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.out = Linear(hidden_size, hidden_size)
        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Transformer_layer(nn.Module):
    """
    transformer layer
    """

    def __init__(self, hidden_size, mlp_dim, dropout_rate, num_heads, attention_dropout_rate, vis):
        super(Transformer_layer, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size,mlp_dim,dropout_rate)
        self.attn = Attention(num_heads, hidden_size, attention_dropout_rate, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x, weights

# hidden_size=256  mlp_dim=1024 num_heads=14  layers=6
class transformer(nn.Module):
    def __init__(self, img_size, num_layers, hidden_size, mlp_dim, dropout_rate,
                 num_heads, attention_dropout_rate, patches=16, vis=False):
        super(transformer, self).__init__()
        self.img_size = (512,320)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.attention_dropout_rate = attention_dropout_rate
        self.vis = vis
        self.dw = True
        self.patches = patches
        if self.patches==8:
            self.h, self.w = 40, 64
        elif self.patches==16:
            self.h, self.w = 20, 32
        else:
            self.h, self.w = 10, 16



        self.patch_size = (self.patches,self.patches)
        self.n_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        self.patch_embeddings = Conv2d(in_channels=3,
                                out_channels=self.hidden_size,
                                kernel_size=self.patch_size,
                                stride=self.patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, self.hidden_size))
        self.dropout = Dropout(self.dropout_rate)

        self.layers = nn.ModuleList([Transformer_layer(self.hidden_size, self.mlp_dim, self.dropout_rate,
                                                       self.num_heads, self.attention_dropout_rate, self.vis) for _ in
                                     range(self.num_layers)])

        # self.conv2d = nn.Conv2d(768, self.out_channels[3], 1)
        # self.up = PatchExpand2D(dim=self.out_channels[3], norm_layer=nn.LayerNorm)

        # self.up = nn.Sequential(nn.Conv2d(768, 512, 1),
        #                    *([nn.UpsamplingBilinear2d(scale_factor=2),
        #                    Conv2dReLU(512, 512, kernel_size=3, stride=1, padding=1)]*self.up_layers))

    def forward(self, x):
        # embedding
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2).transpose(-1, -2).contiguous()  # (B, n_patches, hidden)
        x = x + self.position_embeddings
        x = self.dropout(x)

        for layer in self.layers:
            x, _ = layer(x)

        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        #h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1).view(B, hidden, self.h, self.w)
        # x = self.up(x)
        return x


# # hidden_size=256  mlp_dim=1024 num_heads=14  layers=6
# x = torch.randn(1,3,240,240)
#
# x1 = torch.randn(1,64,128,128)
# x2 = torch.randn(1,128,64,64)
# x3 = torch.randn(1,256,32,32)
# x4 = torch.randn(1,512,16,16)
#
# hidden_size = 256
# dropout_rate = 0.1
# attention_dropout_rate = 0.0
# mlp_dim = 1024
# num_heads = 16
# num_layers = 6
# img_size = _pair(256)
# tf = transformer_fuse(img_size, num_layers, hidden_size, mlp_dim, dropout_rate, num_heads, attention_dropout_rate)
# out = tf([x1,x2,x3,x4])
# for i in out:
#     print(i.shape)


# print(out.shape)
