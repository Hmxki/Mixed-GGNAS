from copy import deepcopy
import math
import os
import torch
import timeit
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from Mixed_GGNAS.cell_module.cells.build_darts_cell import Build_Darts_Cell
# from Mixed_GGNAS.models.mamba_block import MB_BLOCK
from Mixed_GGNAS.utils.early_stopping import EarlyStopping
import torch.optim as optim
from Mixed_GGNAS.cell_module.cells import darts_genotypes
from Mixed_GGNAS.utils.encodings import *
from Mixed_GGNAS.cell_module.buildcells import buildcell


from Mixed_GGNAS.models.vit_block import transformer


class Architecture(object):
    def __init__(self, model, optimizer_arch, criterion):
        self.model = model
        self.arch_optimizer = optimizer_arch
        self.criterion = criterion

    def step(self, input_valid, target_valid):
        self.arch_optimizer.zero_grad()
        logits, mid_outputs, decode_out = self.model(input_valid)
        loss = self.criterion(logits,target_valid)
        loss.backward()
        self.arch_optimizer.step()

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        # lrf=0.001,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9
            # return ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


class CellModel(nn.Module):

    def  __init__(self, chromosome=None, config=None,init_c=3,num_classes=2,base_c=32,device='cuda:0',
                 img_size=(256,256), num_layers=8, hidden_size=512, mlp_dim=768, dropout_rate=0.1,
                 num_heads=8, attention_dropout_rate=0,retrain=False,new_weight=None):
        super(CellModel, self).__init__()
        # CONSTANT
        self.ALPHA = 0.6
        self.BETA = 0.4
        self.NUM_VERTICES = 7
        self.MAX_EDGE_NBR = int(((self.NUM_VERTICES) * (self.NUM_VERTICES - 1)) / 2)

        self.solNo = None
        self.fitness = -1
        self.cost = -1

        self.config = config
        self.chromosome = chromosome
        self.isFeasible = True
        self.device = device
        self.epoch = 0
        self.retrain = retrain
        self.new_w = new_weight

        self.max_nodes = 4
        self._steps = 4
        self._multiplier = 4


        # 单元内部权重
        self.cells_weight = nn.Parameter(torch.ones((8, 6), device=device))
        #self.fuse_w = nn.Parameter(torch.ones(2, device=device))

        self.cells = nn.ModuleList([])
        #self.mp = nn.MaxPool2d((2, 2))

        self.num_classes = num_classes
        self.init_c = init_c
        self.in_channels = [base_c,base_c*2,base_c*4,base_c*8,base_c*16,base_c*8,base_c*4,base_c*2]
        self.out_channels = [base_c*2, base_c*4, base_c*8, base_c*16,base_c*8,base_c*4,base_c*2,base_c]

        # darts channel
        self.c_prev_prev = [base_c, base_c, base_c * 2, base_c * 4, base_c * 8, base_c * 16, base_c * 8, base_c * 4]
        self.c_prev = [base_c, base_c * 2, base_c * 4, base_c * 8, base_c * 16, base_c * 8, base_c * 4, base_c * 2]
        self.c = [base_c * 2 // 4, base_c * 4 // 4, base_c * 8 // 4, base_c * 16 // 4, base_c * 8 // 4, base_c * 4 // 4,
             base_c * 2 // 4, base_c // 4]

        self.stem = nn.Sequential(
            nn.Conv2d(self.init_c, base_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_c),
            nn.ReLU()
        )
        self.compile()

        # mamba
        #self.mamba = MB_BLOCK()

        # transformer
        # img_size, num_layers, hidden_size, mlp_dim, dropout_rate, num_heads, attention_dropout_rate
        self.transformer = transformer(img_size, num_layers, hidden_size, mlp_dim,
                                       dropout_rate, num_heads, attention_dropout_rate).to(device)

        self.trans_out_stem = nn.Conv2d(base_c,num_classes,1)
        #
        self.conv2d_trans = nn.Conv2d(hidden_size, self.out_channels[3], 1)
        # self.conv2d_mamba = nn.Conv2d(768, self.out_channels[3], 1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        #self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1 = []
        for i in range(4):
            temp = nn.Conv2d(self.out_channels[3+i]*2,self.out_channels[3+i],3,padding=1,groups=self.out_channels[3+i]).to(device)
            self.conv1.append(temp)

        self.liner = []
        self.relu_ln = []
        # self.liner1 = []
        # self.relu_ln1 = []
        for i in range(4):
            liner = nn.Linear(self.out_channels[3+i], self.out_channels[4+i], bias=False).to(self.device)
            relu_ln = nn.Sequential(
                nn.ReLU().to(self.device),
                nn.BatchNorm2d(self.out_channels[4+i]).to(self.device)
            )
            self.liner.append(liner)
            self.relu_ln.append(relu_ln)
        # for i in range(4):
        #     liner = nn.Linear(self.out_channels[3+i], self.out_channels[4+i], bias=False).to(self.device)
        #     relu_ln = nn.Sequential(
        #         nn.ReLU().to(self.device),
        #         nn.BatchNorm2d(self.out_channels[4+i]).to(self.device)
        #     )
        #     self.liner1.append(liner)
        #     self.relu_ln1.append(relu_ln)
        self.decode = [nn.Conv2d(self.out_channels[4+i], self.num_classes, kernel_size=1).to(self.device) for i in range(3)]



    def compile(self):
        """ Build U-like Model """

        # create cells
        weight = self.cells_weight.to(self.device)

        for id in range(0,8):
            cell_num = self.config[id]
            # darts 单元
            if cell_num==4:
                print('aaaaaaaaa')
                genotype = eval("darts_genotypes.%s" % 'darts_cell_idrid')
                cell = Build_Darts_Cell(c_prev_prev=self.c_prev_prev[id], c_prev=self.c_prev[id], c=self.c[id], genotype=genotype, cell_type='down' if id<4 else 'up')
            else:
                cell = buildcell(id=id, in_channels=self.in_channels[id], out_channels=self.out_channels[id],
                             cell_num=cell_num, weight=weight[id], retrain=self.retrain,new_w=self.new_w)
            self.cells.append(cell)


        # Output
        self.cells.append(nn.Conv2d(self.out_channels[-1], self.num_classes, kernel_size=1))
        #self.tanh = nn.Tanh()



    def forward(self, inputs):
        # 对原始图像使用vit
        # trans_out = self.transformer(inputs)
        # trans_out = self.conv2d_trans(trans_out)
        # vit_out = trans_out
        # vit_decoder_out = self.decoder(vit_out)
        stem = self.stem(inputs)
        x0 ,x1 = stem, stem
        # encoder
        encode_out = []
        decode_out = []
        count_en = 3
        for i in range(len(self.cells)-1):
            if i<(len(self.cells)-1)//2:
                # 编码器
                x0, x1 = x1, self.cells[i](x1) if self.cells[i].cell_type=='conv' else self.cells[i].forward(x0,x1)
                encode_out.append(x1)
            else:
                # if i == (len(self.cells) - 1) // 2:
                #     # fuse encoder feature
                #     fused_cat = self.fuse(encode_out)
                #     fused_cat = self.fuse_c(fused_cat)
                # 解码器+20\][ewq2][pt5
                # 融合transformer信息
                # cat = torch.cat((x1, trans_out), dim=1)
                # cat = self.conv1[i-4](cat)
                x0, x1, = x1, self.cells[i](x1, encode_out[count_en]) if self.cells[i].cell_type == 'conv' else self.cells[i].forward(x0, x1)
                decode_out.append(x1)
                count_en = count_en-1
                # trans_out = self.up(trans_out)
                # trans_out = self.liner[i-4](trans_out.permute(0,2,3,1)).permute(0,3,1,2)
                # trans_out = self.relu_ln[i-4](trans_out)

                # fused_cat = self.up1(fused_cat)
                # fused_cat = self.liner1[i - 4](fused_cat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                # fused_cat = self.relu_ln1[i - 4](fused_cat)



        # Output
        logits = self.cells[-1](x1)
        for i in range(3):
            decode_out[i] = self.decode[i](decode_out[i])

        return {"out": logits}, [0, encode_out[3]], decode_out


    def evaluate(self, train_loader, val_loader, loss_fn, metric_fn, device):

        # try:
        print(f"Model {self.solNo} Training...")
        self.to(device)  # cuda start

        train_loss = []
        train_dice = []
        train_miou = []
        log = f"Model No: {self.solNo}\n"
        early_stopping = EarlyStopping(patience=80)

        startTime = timeit.default_timer()
        params_to_optimize = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=0.001, weight_decay=5e-5
        )
        lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), 300, warmup=True)

        optimizer_arch = torch.optim.AdamW(
            [self.cells_weight],
            lr=0.0001, weight_decay=5e-6)
        architect = Architecture(self, optimizer_arch=optimizer_arch, criterion=loss_fn)
        smooth_l1loss = nn.SmoothL1Loss()

        min_epoch = self.epoch-1 if self.epoch!=0 else 0

        for epoch in range(min_epoch,80-min_epoch):
            # Train Phase
            self.train()
            train_confmat = metric_fn.ConfusionMatrix(2)
            train_metric_fn_dice = metric_fn.DiceCoefficient(num_classes=2, ignore_index=-100)
            for inputs, labels, mask1, mask2, mask3 in tqdm(train_loader, desc='train'):
                inputs, labels = inputs.to(device), labels.to(device)
                mask1, mask2, mask3 = mask1.to(device), mask2.to(device), mask3.to(device)
                with torch.set_grad_enabled(True):
                    if epoch >= 60 and epoch%5==0:
                        architect.step(inputs, labels)
                    output, mid_outputs, decode_out = self.forward(inputs)
                    #smooth_l1_loss = smooth_l1loss(mid_outputs[0], mid_outputs[1])
                    error = loss_fn(output, labels)
                    decode_loss = loss_fn({'out': decode_out[0]}, mask1) + loss_fn({'out': decode_out[1]},mask2) + loss_fn({'out': decode_out[2]}, mask3)
                    # mask_loss = loss_fn(mask_output, mask_manual)
                    error = error + decode_loss / 3
                    train_loss.append(error.item())
                    train_confmat.update(labels.flatten(), output['out'].argmax(1).flatten())
                    train_metric_fn_dice.update(output['out'], labels)
                    train_dice.append(train_metric_fn_dice.value.item())
                    train_miou.append(train_confmat.compute()[2].mean())
                    optimizer.zero_grad()
                    error.backward()
                    optimizer.step()
                lr_scheduler.step()

                lr = optimizer.param_groups[0]["lr"]
            train_confmat.reduce_from_all_processes()
            train_metric_fn_dice.reduce_from_all_processes()

            torch.cuda.empty_cache()

            val_loss = []
            val_dice = []
            val_miou = []
            self.eval()
            with torch.no_grad():
                val_confmat = metric_fn.ConfusionMatrix(2)
                val_metric_fn_dice = metric_fn.DiceCoefficient(num_classes=2, ignore_index=-255)
                for inputs, labels, mask1, mask2, mask3 in tqdm(val_loader, desc='val'):
                    inputs, labels = inputs.to(device), labels.to(device)
                    mask1, mask2, mask3 = mask1.to(device), mask2.to(device), mask3.to(device)
                    output, mid_outputs, decode_out = self.forward(inputs)
                    #smooth_l1_loss = smooth_l1loss(mid_outputs[0], mid_outputs[1])
                    # loss_sum = 0
                    # for i in range(len(vit_outputs[0])):
                    #     loss_sum += smooth_l1loss(vit_outputs[0][i], vit_outputs[1][i])
                    # cs_loss = CS_loss(vit_outputs[0], vit_outputs[1])
                    error = loss_fn(output, labels)
                    decode_loss = loss_fn({'out': decode_out[0]}, mask1) + loss_fn({'out': decode_out[1]}, mask2) + loss_fn({'out': decode_out[2]}, mask3)
                    error = error + decode_loss / 3
                    val_confmat.update(labels.flatten(), output['out'].argmax(1).flatten())
                    val_metric_fn_dice.update(output['out'], labels)
                    val_dice.append(val_metric_fn_dice.value.item())
                    val_loss.append(error.item())
                    val_miou.append(val_confmat.compute()[2].mean())
                val_confmat.reduce_from_all_processes()
                val_metric_fn_dice.reduce_from_all_processes()

                torch.cuda.empty_cache()

            # Log
            avg_tr_loss = sum(train_loss) / len(train_loss)
            avg_tr_score = sum(train_dice) / len(train_dice)
            avg_tr_miou = sum(train_miou) / len(train_miou)
            avg_val_loss = sum(val_loss) / len(val_loss)
            avg_val_score = sum(val_dice) / len(val_dice)
            avg_val_miou = sum(val_miou) / len(val_miou)
            txt = f"\nEpoch: {epoch}, tr_loss: {avg_tr_loss}, tr_dice_score: {avg_tr_score}, tr_miou: {avg_tr_miou}, " \
                    f"val_loss: {avg_val_loss}, val_dice: {avg_val_score}, val_miou: {avg_val_miou}, lr: {lr}"

            log += txt
            print(txt)

            # Early Stopping Check
            if early_stopping.stopTraining(epoch, avg_val_score, avg_val_miou, avg_tr_score):
                # self.fitness = (self.ALPHA * early_stopping.best_tr_score) + (early_stopping.best_valid_score) + (self.BETA * (1 - ((36 - early_stopping.best_epoch) / 36)))
                self.fitness = early_stopping.best_valid_score
                # self.fitness = (self.ALPHA * (early_stopping.best_valid_score)) + (self.BETA * (1 - ((36 - early_stopping.best_epoch) / 36)))
                self.cost = timeit.default_timer() - startTime
                print(f"Stop Training - Model {self.solNo} , {self.fitness}, {self.cost}")
                break
            self.fitness = early_stopping.best_valid_score
            #self.epoch=self.epoch+1


        # except Exception as e:  # Memory Problems
        #     torch.cuda.empty_cache()
        #     print(e)
        #     return -1, -1

        torch.cuda.empty_cache()

        # self.fitness = (self.ALPHA * early_stopping.best_tr_score) + (early_stopping.best_valid_score) + (self.BETA * (1 - ((36 - early_stopping.best_epoch) / 36)))
        # self.fitness = (self.ALPHA * (early_stopping.best_valid_score)) + (self.BETA * (1 - ((36 - early_stopping.best_epoch) / 36)))
        self.cost = timeit.default_timer() - startTime
        log += f"\nElapsed Time: {self.cost}, Fitness: {self.fitness}, Best Valid: {early_stopping.best_valid_score}, Best TR: {early_stopping.best_tr_score}"
        with open(f"../results/idrid/model_{self.solNo}.txt", "w") as f:
            f.write(log)


        return self.fitness, self.cost



    def get_neighborhood(self, nbr_ops, CELLS, FILTERS, neighbor_rnd, shuffle=True):
        nbhd = []
        # add op neighbors
        for vertex in range(self.NUM_VERTICES - 2):
            available = [op for op in range(nbr_ops) if op != self.org_ops[vertex]]
            for op in available:
                new_matrix = deepcopy(self.org_matrix)
                new_ops = deepcopy(self.org_ops)
                new_ops[vertex] = op
                new_arch = {'matrix': new_matrix, 'ops': new_ops, 'nbr_cell': self.nbr_cell,
                            'init_filter': self.nbr_filters}
                nbhd.append(new_arch)

        # add edge neighbors
        for src in range(0, self.NUM_VERTICES - 1):
            for dst in range(src + 1, self.NUM_VERTICES):
                new_matrix = deepcopy(self.org_matrix)
                new_ops = deepcopy(self.org_ops)
                new_matrix[src][dst] = 1 - new_matrix[src][dst]
                new_arch = {'matrix': new_matrix, 'ops': new_ops, 'nbr_cell': self.nbr_cell,
                            'init_filter': self.nbr_filters}
                nbhd.append(new_arch)

                # add nbr_cell neighbors
        available = [nbr_cell for nbr_cell in CELLS if nbr_cell != self.nbr_cell]
        for nbr_cell in available:
            new_matrix = deepcopy(self.org_matrix)
            new_ops = deepcopy(self.org_ops)
            new_arch = {'matrix': new_matrix, 'ops': new_ops, 'nbr_cell': nbr_cell, 'init_filter': self.nbr_filters}
            nbhd.append(new_arch)

        # add nbr_filter neighbors
        available = [nbr_filter for nbr_filter in FILTERS if nbr_filter != self.nbr_filters]
        for nbr_filter in available:
            new_matrix = deepcopy(self.org_matrix)
            new_ops = deepcopy(self.org_ops)
            new_arch = {'matrix': new_matrix, 'ops': new_ops, 'nbr_cell': self.nbr_cell, 'init_filter': nbr_filter}
            nbhd.append(new_arch)

        if shuffle:
            neighbor_rnd.shuffle(nbhd)
        return nbhd

    def reset(self):
        for param in self.parameters():
            param.requires_grad = True
            if len(param.shape) > 1:
                torch.nn.init.xavier_uniform_(param)
                param.data.grad = None

    def encode(self, predictor_encoding):

        if predictor_encoding == 'path':
            return encode_paths(self.get_path_indices())
        elif predictor_encoding == 'caz':
            ops =deepcopy(self.org_ops)
            ops = [OPS[i] for i in ops]
            ops.insert(0, 'input')
            ops.append('output')
            return encode_caz(self.org_matrix, ops)

    def get_paths(self):
        """
        return all paths from input to output
        """
        ops = deepcopy(self.org_ops)
        ops = [OPS[i] for i in ops]
        ops.insert(0, 'input')
        ops.append('output')
        paths = []
        for j in range(0, self.NUM_VERTICES):
            paths.append([[]]) if self.org_matrix[0][j] else paths.append([])

        # create paths sequentially
        for i in range(1, self.NUM_VERTICES - 1):
            for j in range(1, self.NUM_VERTICES):
                if self.org_matrix[i][j]:
                    for path in paths[i]:
                        paths[j].append([*path, ops[i]])
        return paths[-1]

    def get_path_indices(self):
        """
        compute the index of each path
        There are 9^0 + ... + 9^5 paths total.
        (Paths can be length 0 to 5, and for each path, for each node, there
        are nine choices for the operation.)
        """
        paths = self.get_paths()
        ops = OPS
        mapping = {op: idx for idx, op in enumerate(OPS)}

        path_indices = []

        for path in paths:
            index = 0
            for i in range(self.NUM_VERTICES - 1):
                if i == len(path):
                    path_indices.append(index)
                    break
                else:
                    index += len(ops) ** i * (mapping[path[i]] + 1)

        path_indices.sort()
        return tuple(path_indices)


#
# if __name__ == "__main__":
#     model = CellModel([], [4,4,4,4,4,4,4,4], init_c=3, device='cuda', img_size=(240,240))
#     model.to("cuda")
#     x = torch.randn(1, 3, 240, 240).to('cuda')
#     out = model(x)
#     print(out['out'].shape)