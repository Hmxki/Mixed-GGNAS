import pickle
import os
import numpy as np
import torch
from torch import nn
from vit_cells_model import vit_CellModel
import UNAS_Net.utils.distributed_utils as utils_
from UNAS_Net.utils.losses import criterion

from UNAS_Net.utils.data_transform import get_transform
from UNAS_Net.utils.my_dataset import DriveDataset

def readPickleFile(file):
    with open(f"../best_model/model_{file}_idrid.pkl", "rb") as f:
        data = pickle.load(f)

    return data
class vit_search():
    def __init__(self,modelNo,datapath,crop_size,std,mean,batch_size,
                 pop_size,mutation_factor=0.5,crossover_prob = 0.5,init_c=3,
                 device='cuda',img_size=(240,240)):
        # 读取搜索到的最优模型
        self.modelNo = modelNo
        self.datapath = datapath
        self.crop_size = crop_size
        self.std = std
        self.mean = mean
        self.batch_size = batch_size
        self.init_c = init_c
        self.device = device
        self.img_size = img_size

        self.pop_size = pop_size  # 20
        self.mutation_factor = mutation_factor  # 0.5
        self.crossover_prob = crossover_prob

        self.population = []
        self.allModels = dict()
        self.best_arch = None
        self.best_model_numb = -1
        self.seed = 42  # 42
        self.pop_solNo = []

        self.MAX_SOL = 100
        self.DIMENSIONS = 5

        self.train_dataloader, self.val_dataloader = self.data_load()
        self.loss_fn = criterion()
        self.metric_fn = utils_


        self.model = readPickleFile(self.modelNo)
        self.model_config = self.model.config
        # 确定选取单元
        self.w_tensor = torch.empty(8,6)
        for index, cell_w in enumerate(self.model.cells_weight):
            no_one = cell_w == 1
            new_tensor = cell_w.clone()
            new_tensor[no_one] = 0
            max_index = torch.argmax(new_tensor)
            new_tensor[max_index] = 1.0
            no_one = new_tensor != 1
            new_tensor[no_one] = 0
            self.w_tensor[index] = new_tensor
        # self.model.cells_weight = nn.ParameterList(
        #     [nn.Parameter(w_tensor[i]) for i in range(8)].to('cuda')
        # )

    def reset(self):
        self.best_model_numb = -1
        self.best_arch = None
        self.population = []
        self.allModels = dict()
        self.init_rnd_nbr_generators()

    def writePickle(self, data, name):
        # Write History
        with open("../models/model_{}.pkl".format(name), "wb") as pkl:
            pickle.dump(data, pkl)


    def data_load(self):
        train_dataset = DriveDataset(self.datapath,
                                     train=True,
                                     transforms=get_transform(train=True, crop_size=self.crop_size, mean=self.mean, std=self.std))

        val_dataset = DriveDataset(self.datapath,
                                   train=False,
                                   transforms=get_transform(train=False, crop_size=self.crop_size, mean=self.mean, std=self.std))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.batch_size,
                                                   num_workers=0,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1,
                                                 num_workers=0,
                                                 pin_memory=True,
                                                 collate_fn=val_dataset.collate_fn)
        return train_loader,val_loader

    def checkSolution(self, config):
        for i in self.allModels.keys():
            config_2 = self.allModels[i]
            if np.array_equal(config_2, config):
                return True, config_2
            else:
                continue
        return False, None

    def init_rnd_nbr_generators(self):
        # Random Number Generators
        self.crossover_rnd = np.random.RandomState(self.seed)
        self.sample_pop_rnd = np.random.RandomState(self.seed)
        self.init_pop_rnd = np.random.RandomState(self.seed)

    def get_param_value(self, value, step_size):
        # 创建离散区间
        ranges = np.arange(start=0, stop=1, step=1 / step_size)
        # 返回大于value的
        return np.where((value < ranges) == False)[0][-1]

    def vector_to_config(self, vector):
        '''Converts numpy array to discrete values'''

        try:
            # 创建一个初始值为0的np数组
            config = np.zeros(self.DIMENSIONS, dtype='uint8')

            # dim 4
            config[0] = self.get_param_value(vector[0], 4)
            # patch_size
            config[1] = self.get_param_value(vector[1], 3)
            # heads
            config[2] = self.get_param_value(vector[2], 4)
            # mlp_sim
            config[3] = self.get_param_value(vector[3], 4)
            # depth
            config[4] = self.get_param_value(vector[4], 4)



        except:
            print("HATA...", vector)

        return config

    def init_population(self, pop_size=None):
        i = 0
        while i < pop_size:
            # 创建一个染色体，长度为5
            chromosome = self.init_pop_rnd.uniform(low=0.0, high=1.0, size=self.DIMENSIONS)
            # 计算np数组转为离散数据
            config = self.vector_to_config(chromosome)
            # 根据配置信息创建种群个体模型
            # init_c=3,num_classes=2,base_c=32
            model = vit_CellModel(chromosome, self.model_config, config, w_tensor=self.w_tensor, init_c=self.init_c,device = self.device,img_size=self.img_size)
            # Same Solution Check
            isSame, _ = self.checkSolution(model.config_vit)
            if not isSame:
                model.solNo = self.solNo
                self. solNo = 1 + self.solNo
                self.population.append(model)
                self.allModels[model.solNo] = list(model.config_vit)
                self.writePickle(model, model.solNo)
                print('No.'+str(i)+'finished!')
                i = 1 + i

        return np.array(self.population)

    def f_objective(self, model):
        if model.isFeasible == False:  # Feasibility Check
            return -1, -1

        # Else
        fitness, cost = model.evaluate(self.train_dataloader, self.val_dataloader, self.loss_fn, self.metric_fn, self.device)
        if fitness != -1:
            self.totalTrainedModel = 1 + self.totalTrainedModel
        return fitness, cost

    def init_eval_pop(self):
        '''
            Creates new population of 'pop_size' and evaluates individuals.
        '''
        print("Start Initialization...")
        # 初始化一个种群
        self.population = self.init_population(self.pop_size)
        self.pop_solNo=[]
        for i in self.population:
            self.pop_solNo.append(i.solNo)
        self.best_arch = self.population[0]
        self.best_model_numb = self.population[0].solNo

        for i in range(self.pop_size):
            model = self.population[i]
            # 获取模型适应度
            model.fitness, cost = self.f_objective(model)
            # 保存新模型
            self.writePickle(model, model.solNo)
            # 如果当前模型的适应厚比最优模型适应度高，则更新最优模型
            if model.fitness >= self.best_arch.fitness:
                self.best_arch = model


    def run(self):
        self.solNo = 0
        self.generation = 0
        self.totalTrainedModel = 0
        self.reset()
        # 获取种群初始适应度
        self.init_eval_pop()
        # 训练模型总数self.MAX_SOL=500
        while self.totalTrainedModel < self.MAX_SOL:
            self.evolve_generation()
            self.check_point = False
            print(f"Generation:{self.generation}, Best: {self.best_arch.fitness}, {self.best_arch.solNo}")
            self.generation = 1 + self.generation
            self.log_dic['generation'] = self.generation
            with open('../log/check_point.txt', 'w') as f:
                f.write(str(self.log_dic))

if __name__ == "__main__":
    modelNo = 14
    datapath = ''
    crop_size = 240
    mean = (0.402, 0.270, 0.184)
    std = (0.298, 0.204, 0.138)
    batch_size = 1
    pop_size = 3
    de = vit_search(modelNo,datapath,crop_size,std,mean,batch_size,
                 pop_size,mutation_factor=0.5,crossover_prob=0.5, img_size=(240,240))
    de.run()