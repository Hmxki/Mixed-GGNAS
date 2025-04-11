from torch import nn
from Mixed_GGNAS.cell_module.cells.prim_ops_set import *


class Build_Darts_Cell(nn.Module):
    """Build a cell from genotype"""

    def __init__(self, genotype, c_prev_prev, c_prev, c, cell_type, dropout_prob=0):
        super(Build_Darts_Cell, self).__init__()
        self.cell_type = 'darts'

        if cell_type == 'down':
            # Note: the s0 size is twice than s1!
            self.preprocess0 = ConvOps(c_prev_prev, c, kernel_size=1, stride=2, ops_order='act_weight_norm')
        else:
            self.preprocess0 = ConvOps(c_prev_prev, c, kernel_size=1, ops_order='act_weight_norm')
        self.preprocess1 = ConvOps(c_prev, c, kernel_size=1, ops_order='act_weight_norm')

        if cell_type == 'up':
            op_names, idx = zip(*genotype.up)
            concat = genotype.up_concat
        else:
            op_names, idx = zip(*genotype.down)
            concat = genotype.down_concat
        self.dropout_prob = dropout_prob
        self._compile(c, op_names, idx, concat)

    def _compile(self, c, op_names, idx, concat):
        assert len(op_names) == len(idx)
        self._num_meta_node = len(op_names) // 2
        self._concat = concat
        self._multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, idx):
            op = OPS[name](c, None, affine=True, dp=self.dropout_prob)
            self._ops += [op]
        self._indices = idx

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._num_meta_node):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]

            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]

            h1 = op1(h1)
            h2 = op2(h2)

            # the size of h1 and h2 may be different, so we need interpolate
            if h1.size() != h2.size() :
                _, _, height1, width1 = h1.size()
                _, _, height2, width2 = h2.size()
                if height1 > height2 or width1 > width2:
                    h2 = interpolate(h2, (height1, width1))
                else:
                    h1 = interpolate(h1, (height2, width2))
            s = h1+h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


# if __name__=="__main__":
#     # 随机生成输入张量数据
#     x1 = torch.randn(2, 16, 480, 480)
#     x2 = torch.randn(2, 32, 240, 240)
#     x3 = torch.randn(2, 64, 120, 120)
#     x4 = torch.randn(2, 128, 60, 60)
#     x5 = torch.randn(2, 256, 30, 30)
#     x6 = torch.randn(2, 32, 240, 240)
#     x7 = torch.randn(2, 16, 480, 480)
#
#     base_c = 16
#     in_channels = [base_c, base_c * 2, base_c * 4, base_c * 8, base_c * 16, base_c * 8, base_c * 4, base_c * 2]
#     out_channels = [base_c * 2, base_c * 4, base_c * 8, base_c * 16, base_c * 8, base_c * 4, base_c * 2, base_c]
#
#     c_prev_prev = [base_c,base_c,base_c*2,base_c * 4,    base_c * 8,  base_c * 16, base_c * 8, base_c * 4]
#     c_prev = [base_c,base_c*2,base_c*4,base_c * 8,       base_c * 16, base_c * 8,  base_c * 4, base_c * 2]
#     c = [base_c*2//4, base_c * 4//4, base_c * 8//4, base_c * 16//4, base_c * 8//4, base_c * 4//4, base_c * 2//4, base_c//4]
#
#     genotype = eval("genotypes.%s" % 'DARTS')
#     cells = []
#     # c=下一个单元输入通道的1/4
#     for i in range(4):
#         cell = Build_Darts_Cell(genotype, c_prev_prev=c_prev_prev[i], c_prev=c_prev[i], c=c[i], cell_type='down')
#         cells.append(cell)
#     for i in range(4):
#         cell = Build_Darts_Cell(genotype, c_prev_prev=c_prev_prev[i+4], c_prev=c_prev[i+4], c=c[i+4], cell_type='up')
#         cells.append(cell)
#
#
#     input1, input2 = x1, x1
#     for cell_ in cells:
#         #print('input1',input1.shape,'input2',input2.shape)
#         input1, input2 = input2, cell_(input1,input2)
#         print('cell'+'{}_out'.format(cells.index(cell_)),input2.shape)
#         print('')




# 记录每一个单元的通道信息，然后如果当前单元使darts单元，获取前一个单元的输出通道，后一个单元的输入通道