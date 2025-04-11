import timeit
import random
import numpy as np
import torch
from thop import profile
from torch import optim
from tqdm import tqdm
from UNAS_Net.models.cells_model import CellModel,Architecture
from UNAS_Net.utils.early_stopping import EarlyStopping
from UNAS_Net.utils.save_best_model import BestModelCheckPoint
from vit_cells_model import vit_CellModel
from UNAS_Net.utils.my_dataset import DriveDataset
from UNAS_Net.utils.data_transform import get_transform
from UNAS_Net.utils.losses import criterion, CS_loss,uncertainty_loss
from UNAS_Net.utils.metrics import *
import pickle
import UNAS_Net.utils.distributed_utils as utils_

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


def main(args):

    crop_size = 0
    # using compute_mean_std.py
    mean = (0.457, 0.221, 0.064)
    std = (0.321, 0.167, 0.086)
    # mean = (0.485, 0.341, 0.628)
    # std = (0.231, 0.237, 0.193)
    # mean = (0.709, 0.381, 0.224)
    # std = (0.127, 0.079, 0.043)
    # mean = 0.329
    # std = 0.223
    batch_size = args.batch_size
    seed=42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda:{}'.format(0))



    train_dataset = DriveDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, crop_size=crop_size, mean=mean, std=std))
    # train_dataset.img_list = train_dataset.img_list[:48]
    # train_dataset.manual = train_dataset.manual[:48]

    val_dataset = DriveDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, crop_size=crop_size, mean=mean, std=std))
    # val_dataset.img_list = train_dataset.img_list[:8]
    # val_dataset.manual = train_dataset.manual[:8]

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=0,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             num_workers=0,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    loss_fn = criterion()
    metric_fn = utils_
    smooth_l1loss = nn.SmoothL1Loss()

    #model = UNet(in_channels=1,num_classes=2)

    with open(f"../best_model/model_5.pkl", "rb") as f:
        best_model = pickle.load(f)
    k_index = []
    for w_list in best_model.cells_weight:
        k_index.append(list(w_list).index(max(w_list)))

    # for i in range(len(best_model.config)):
    #     best_model.config[i] = 4
    model = vit_CellModel(best_model.chromosome, best_model.config, best_model.config_vit, new_weight=k_index,
                          init_c=3, device=device, img_size=(crop_size,crop_size), retrain=True)
    model.to(device)
    input = torch.randn(4, 3, 320, 512).to(device)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

    del best_model



    train_loss = []
    train_dice = []
    train_miou = []
    log = f"Model : {'denas_vitde_idrid_wo_darts'}\n"
    early_stopping = EarlyStopping(patience=300)
    startTime = timeit.default_timer()
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=0.001, weight_decay=5e-5
    )
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), 300, warmup=True)

    # optimizer_arch = torch.optim.AdamW(
    #     [model.cells_weight],
    #     lr=0.0001, weight_decay=5e-6
    # )
    # architect = Architecture(model, optimizer_arch=optimizer_arch,
    #                          criterion=loss_fn)

    checkpoint = BestModelCheckPoint('denas_vitde_idrid_wo_darts_',args.count)

    for epoch in range(300):
        model.train()
        # mask_output = {}
        train_confmat = metric_fn.ConfusionMatrix(2)
        train_metric_fn_dice = metric_fn.DiceCoefficient(num_classes=2, ignore_index=-100)
        for inputs, labels, mask1, mask2, mask3 in tqdm(train_loader, desc='train'):
            inputs, labels = inputs.to(device), labels.to(device)
            mask1, mask2, mask3 = mask1.to(device), mask2.to(device), mask3.to(device)
            with torch.set_grad_enabled(True):
                # if epoch >= 60 and epoch % 10 == 0:
                #     architect.step(inputs, labels)
                output, mid_outputs, decode_out = model.forward(inputs)
                smooth_l1_loss = smooth_l1loss(mid_outputs[0],mid_outputs[1])
                error = loss_fn(output, labels)
                decode_loss = loss_fn({'out': decode_out[0]}, mask1) + loss_fn({'out': decode_out[1]}, mask2) + loss_fn({'out': decode_out[2]}, mask3)
                error = error+smooth_l1_loss+decode_loss/3
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
        model.eval()
        # with torch.no_grad():
        val_confmat = metric_fn.ConfusionMatrix(2)
        val_metric_fn_dice = metric_fn.DiceCoefficient(num_classes=2, ignore_index=-255)
        for inputs, labels, mask1, mask2, mask3 in tqdm(val_loader, desc='val'):
            inputs, labels = inputs.to(device), labels.to(device)
            mask1, mask2, mask3 = mask1.to(device), mask2.to(device), mask3.to(device)
            output, mid_outputs, decode_out = model.forward(inputs)
            smooth_l1_loss = smooth_l1loss(mid_outputs[0],mid_outputs[1])

            error = loss_fn(output, labels)
            decode_loss = loss_fn({'out': decode_out[0]}, mask1) + loss_fn({'out': decode_out[1]}, mask2) + loss_fn({'out': decode_out[2]}, mask3)
            error = error+smooth_l1_loss+decode_loss/3
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
        with open(f"H:\\hmx\\NAS_data\\cvc\\results\\model_denas_vitde_idrid_wo_darts_{args.count}.txt", "w") as f:
            f.write(log)

        # Early Stopping Check
        if early_stopping.stopTraining(epoch, avg_val_score, avg_val_miou, avg_tr_score):
            fitness = early_stopping.best_valid_score
            cost = timeit.default_timer() - startTime
            print(f"Stop Training - Model {args.modelNo} , {fitness}, {cost}")
            break
        model.fitness = early_stopping.best_valid_score

        checkpoint.check(avg_val_miou, model, seed)

    cost = timeit.default_timer() - startTime
    log += f"\nElapsed Time: {cost}, Fitness: {model.fitness}, Best Valid: {early_stopping.best_valid_score}, Best TR: {early_stopping.best_tr_score}"
    with open(f"H:\\hmx\\NAS_data\\cvc\\results\\model_denas_vitde_idrid_wo_darts_{args.count}.txt", "w") as f:
        f.write(log)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="", help="DRIVE root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--modelNo", default=20, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=400, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--count', default=1, type=int, help='')

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_args()
    for i in range(1,20):
        args.count = i
        main(args)