import ast
import os

import numpy as np
import pickle

from Mixed_GGNAS.models.cells_model import CellModel
from Mixed_GGNAS.utils.distances import *


def readPickleFile(file):
    with open(f"H:\\hmx\\NAS_data\\idrid\\models_de\\model_{file}.pkl", "rb") as f:
        data = pickle.load(f)

    return data

def readPickleFile_trials(file):
    with open(f"H:\\hmx\\NAS_data\\idrid\\trials_de\\model_{file}.pkl", "rb") as f:
        data = pickle.load(f)

    return data

class MYDE():

    def __init__(self, train_dataloader,
                 val_dataloader,
                 loss_fn,
                 metric_fn,
                 device,
                 img_size,
                 pop_size=None,
                 mutation_factor=None,
                 crossover_prob=None,
                 boundary_fix_type='random',
                 seed=None,
                 mutation_strategy='rand1',
                 crossover_strategy='bin',
                 init_c=3
                 ):
        # 初始化代码


        # DE related variables
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.device = device
        self.init_c = init_c
        self.img_size = img_size

        self.pop_size = pop_size  # 20
        self.mutation_factor = mutation_factor  # 0.5
        self.crossover_prob = crossover_prob  # 0.5
        self.mutation_strategy = mutation_strategy  # rand1
        self.crossover_strategy = crossover_strategy  # bin
        self.boundary_fix_type = boundary_fix_type  # random

        # Global trackers
        self.population = []
        self.history = []
        self.allModels = dict()
        self.best_arch = None
        self.best_model_numb = -1
        self.seed = seed  # 42
        self.pop_solNo = []

        # CONSTANTS
        self.MAX_SOL = 100
        self.NUM_EDGES = 9
        self.NUM_VERTICES = 7
        self.DIMENSIONS = 8
        self.MAX_NUM_CELL = 5
        self.log_dic = {}
        self.model_numb=0

    def reset(self):
        self.best_model_numb = -1
        self.best_arch = None
        self.population = []
        self.allModels = dict()
        self.history = []
        self.init_rnd_nbr_generators()

    def init_rnd_nbr_generators(self):
        # Random Number Generators
        self.crossover_rnd = np.random.RandomState(self.seed)
        self.sample_pop_rnd = np.random.RandomState(self.seed)
        self.init_pop_rnd = np.random.RandomState(self.seed)

    def writePickle(self, data, name):
        # Write History
        with open("H:\\hmx\\NAS_data\\idrid\\models_de\\model_{}.pkl".format(name), "wb") as pkl:
            pickle.dump(data, pkl)

    def writePickle_de(self, data, name):
        # Write History
        with open("H:\\hmx\\NAS_data\\idrid\\trials_de\\model_{}.pkl".format(name), "wb") as pkl:
            pickle.dump(data, pkl)

    # Initialize population
    def init_population(self, pop_size=None):
        i = 0
        while i < pop_size:
            # 创建一个染色体，长度为32
            chromosome = self.init_pop_rnd.uniform(low=0.0, high=1.0, size=self.DIMENSIONS)
            # 计算np数组转为离散数据
            config = self.vector_to_config(chromosome)
            # 根据配置信息创建种群个体模型
            # init_c=3,num_classes=2,base_c=32
            model = CellModel(chromosome, config, init_c=self.init_c,device = self.device,img_size=self.img_size)
            # Same Solution Check
            isSame, _ = self.checkSolution(model.config)
            if not isSame:
                model.solNo = self.solNo
                self.solNo = 1 + self.solNo
                self.population.append(model)
                self.allModels[model.solNo] = list(model.config)
                self.writePickle(model, model.solNo)
                print('No.'+str(i)+'finished!')
                i = 1 + i

        return np.array(self.population)

    def sample_population(self, size=None):
        '''Samples 'size' individuals'''

        selection = self.sample_pop_rnd.choice(np.arange(len(self.population)), size, replace=False)
        return self.population[selection]

    def boundary_check(self, vector):
        '''
        Checks whether each of the dimensions of the input vector are within [0, 1].
        If not, values of those dimensions are replaced with the type of fix selected.

        projection == The invalid value is truncated to the nearest limit
        random == The invalid value is repaired by computing a random number between its established limits
        reflection == The invalid value by computing the scaled difference of the exceeded bound multiplied by two minus

        '''
        violations = np.where((vector > 1) | (vector < 0))[0]
        if len(violations) == 0:
            return vector

        if self.boundary_fix_type == 'projection':
            vector = np.clip(vector, 0.0, 1.0)
        elif self.boundary_fix_type == 'random':
            vector[violations] = np.random.uniform(low=0.0, high=1.0, size=len(violations))
        elif self.boundary_fix_type == 'reflection':
            vector[violations] = [0 - v if v < 0 else 2 - v if v > 1 else v for v in vector[violations]]

        return vector

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

            for i in range(8):
                # 获取单元离散值
                config[i] = self.get_param_value(vector[i], 4)

            # for i in range(8, 16):
            #     # 获取卷积核离散值
            #     config[i] = self.get_param_value(vector[i], 3)
            #
            # for i in range(16, 24):
            #     # 获取跳跃连接离散值
            #     config[i] = self.get_param_value(vector[i], 2)
            #
            # for i in range(24, 32):
            #     # 获取注意力门控离散值
            #     config[i] = self.get_param_value(vector[i], 2)

        except:
            print("HATA...", vector)

        return config

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
                self.best_model_numb = model.solNo
            self.log_dic['solNo'] = model.solNo
            self.log_dic['allModels'] = self.allModels
            self.log_dic['best_model_numb'] = self.best_model_numb
            self.log_dic['pop_solNo'] = list(self.pop_solNo)
            with open('../log/check_point_idrid.txt', 'w') as f:
                f.write(str(self.log_dic))

    def mutation_rand1(self, r1, r2, r3):
        '''Performs the 'rand1' type of DE mutation
        '''
        diff = r2 - r3
        mutant = r1 + self.mutation_factor * diff
        return mutant

    def mutation_rand2(self, r1, r2, r3, r4, r5):
        '''Performs the 'rand2' type of DE mutation
        '''
        diff1 = r2 - r3
        diff2 = r4 - r5
        mutant = r1 + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation_currenttobest1(self, current, best, r1, r2):
        diff1 = best - current
        diff2 = r1 - r2
        mutant = current + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation(self, current=None, best=None):
        '''Performs DE mutation
        '''
        if self.mutation_strategy == 'rand1':
            # 随机选择3个个体
            r1, r2, r3 = self.sample_population(size=3)
            # 使用选择的3个个体进行变异操作
            mutant = self.mutation_rand1(r1.chromosome, r2.chromosome, r3.chromosome)

        elif self.mutation_strategy == 'rand2':
            r1, r2, r3, r4, r5 = self.sample_population(size=5)
            mutant = self.mutation_rand2(r1.chromosome, r2.chromosome, r3.chromosome, r4.chromosome, r5.chromosome)

        elif self.mutation_strategy == 'best1':
            r1, r2 = self.sample_population(size=2)
            mutant = self.mutation_rand1(best, r1.chromosome, r2.chromosome)

        elif self.mutation_strategy == 'best2':
            r1, r2, r3, r4 = self.sample_population(size=4)
            mutant = self.mutation_rand2(best, r1.chromosome, r2.chromosome, r3.chromosome, r4.chromosome)

        elif self.mutation_strategy == 'currenttobest1':
            r1, r2 = self.sample_population(size=2)
            mutant = self.mutation_currenttobest1(current, best.chromosome, r1.chromosome, r2.chromosome)

        elif self.mutation_strategy == 'randtobest1':
            r1, r2, r3 = self.sample_population(size=3)
            mutant = self.mutation_currenttobest1(r1.chromosome, best.chromosome, r2.chromosome, r3.chromosome)

        return mutant

    def crossover_bin(self, target, mutant):
        '''Performs the binomial crossover of DE
        '''
        cross_points = self.crossover_rnd.rand(self.DIMENSIONS) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[self.crossover_rnd.randint(0, self.DIMENSIONS)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def crossover_exp(self, target, mutant):
        '''
            Performs the exponential crossover of DE
        '''
        n = self.crossover_rnd.randint(0, self.DIMENSIONS)
        L = 0
        while ((self.crossover_rnd.rand() < self.crossover_prob) and L < self.DIMENSIONS):
            idx = (n + L) % self.DIMENSIONS
            target[idx] = mutant[idx]
            L = L + 1
        return target

    def crossover(self, target, mutant):
        '''
            Performs DE crossover
        '''
        if self.crossover_strategy == 'bin':
            offspring = self.crossover_bin(target, mutant)
        elif self.crossover_strategy == 'exp':
            offspring = self.crossover_exp(target, mutant)
        return offspring

    # def readPickleFile(self, file):
    #     with open(f"results/model_{file}.pkl", "rb") as f:
    #         data = pickle.load(f)
    #
    #     return data

    def checkSolution(self, config):
        for i in self.allModels.keys():
            config_2 = self.allModels[i]
            if np.array_equal(config_2, config):
                return True, config_2
            else:
                continue
        return False, None


    def evolve_generation(self):
        '''
            Performs a complete DE evolution: mutation -> crossover -> selection
        '''
        trials = []
        Pnext = []  # Next population
        # 把断点之前选择的模型加入Pnext
        if 'Pnext' in self.log_dic.keys():
            for i in self.log_dic['Pnext']:
                temp = readPickleFile(i)
                Pnext.append(temp)

        # generationBest = max(self.population, key=lambda x: x.fitness)
        generationBest = None
        # mutation -> crossover
        # 如果是断点，读取断点处保存的进化个体
        if self.check_point and len(os.listdir('H:\\hmx\\NAS_data\\idrid\\trials_de'))!=0:
            for i in range(self.pop_size):
                # 读取断点处的所有进化个体
                model_temp = readPickleFile_trials(i)
                trials.append(model_temp)
        else:
            for j in range(self.pop_size):
                target = self.population[j].chromosome
                # 变异
                mutant = self.mutation(current=target, best=generationBest)
                # 交叉
                trial = self.crossover(target, mutant)
                # 边界检查
                trial = self.boundary_check(trial)
                # 计算np数组转为离散数据
                config = self.vector_to_config(trial)
                model = CellModel(trial, config, init_c=self.init_c,img_size=self.img_size)
                model.solNo = self.solNo
                self.solNo = 1 + self.solNo
                trials.append(model)
                self.writePickle_de(model, j)

        trials = np.array(trials)

        # selection
        for j in range(self.model_numb%self.pop_size,self.pop_size):
            target = self.population[j]
            mutant = trials[j]

            isSameSolution, sol = self.checkSolution(mutant.config)
            if isSameSolution:
                print("SAME SOLUTION",sol)
                self.writePickle(mutant, mutant.solNo)
                self.allModels[mutant.solNo] = list(mutant.config)
                self.log_dic['solNo'] = mutant.solNo
                self.log_dic['generation'] = self.generation
                self.log_dic['allModels'] = self.allModels
                self.log_dic['best_model_numb'] = self.best_model_numb
                with open('../log/check_point_idrid.txt', 'w') as f:
                    f.write(str(self.log_dic))
            else:
                self.f_objective(mutant)
                self.writePickle(mutant, mutant.solNo)
                self.allModels[mutant.solNo] = list(mutant.config)

                self.log_dic['solNo'] = mutant.solNo
                self.log_dic['generation'] = self.generation
                self.log_dic['allModels'] = self.allModels
                self.log_dic['best_model_numb'] = self.best_model_numb
                with open('../log/check_point_idrid.txt', 'w') as f:
                    f.write(str(self.log_dic))

            # Check Termination Condition
            if self.totalTrainedModel > self.MAX_SOL:
                return
            #######

            if mutant.fitness >= target.fitness:
                Pnext.append(mutant)
                # Best Solution Check
                if mutant.fitness >= self.best_arch.fitness:
                    self.best_arch = mutant
                    self.best_model_numb = mutant.solNo
                self.log_dic['Pnext'] = [i.solNo for i in Pnext]
                with open('../log/check_point_idrid.txt', 'w') as f:
                    f.write(str(self.log_dic))
            else:
                Pnext.append(target)
                self.log_dic['Pnext'] = [i.solNo for i in Pnext]
                with open('../log/check_point_idrid.txt', 'w') as f:
                    f.write(str(self.log_dic))
        self.model_numb=0


        self.population = np.array(Pnext)
        # 更新种群之后清除记录
        self.log_dic['Pnext'] = []
        with open('../log/check_point_idrid.txt', 'w') as f:
            f.write(str(self.log_dic))
        self.pop_solNo=[]
        for i in self.population:
            self.pop_solNo.append(i.solNo)
        self.log_dic['pop_solNo'] = list(self.pop_solNo)
        with open('../log/check_point_idrid.txt', 'w') as f:
            f.write(str(self.log_dic))

    def run(self, seed, check_point, pop_size):
        self.check_point = check_point
        if self.check_point and len(os.listdir('H:\\hmx\\NAS_data\\idrid\\models_de'))!=0:
            self.reset()
            f = open('../log/check_point_idrid.txt', 'r')
            self.log_dic = ast.literal_eval(f.read())
            self.model_numb = self.log_dic['solNo'] + 1
            if self.model_numb<=pop_size:
                self.best_model_numb = self.log_dic['best_model_numb']
                self.best_arch = readPickleFile(self.best_model_numb)
                self.pop_size = pop_size
                self.allModels = self.log_dic['allModels']
                # 断点模型序号

                self.totalTrainedModel = self.log_dic['solNo']
                pop_array = []
                for i in range(pop_size):
                    model_temp = readPickleFile(i)
                    pop_array.append(model_temp)
                self.population = np.array(pop_array)
                del pop_array
                # 判断是否是初始种群中的模型
                # 如果是初始化种群中的模型，则完成初始化种群训练，并进化
                for i in range(pop_size-(self.log_dic['solNo']+1)):
                    model = readPickleFile(self.model_numb)
                    model.reset()
                    # 获取模型适应度
                    model.fitness, cost = self.f_objective(model)
                    # 保存新模型
                    self.writePickle(model, model.solNo)
                    # 如果当前模型的适应厚比最优模型适应度高，则更新最优模型
                    if model.fitness >= self.best_arch.fitness:
                        self.best_arch = model
                        self.best_model_numb = model.solNo
                    self.model_numb = self.model_numb+1
                    self.log_dic['solNo'] = model.solNo
                    self.log_dic['allModels'] = self.allModels
                    self.log_dic['best_model_numb'] = self.best_model_numb
                    with open('../log/check_point_idrid.txt', 'w') as f:
                        f.write(str(self.log_dic))

                # 初始种群训练完毕，开始进化
                self.solNo = pop_size
                self.generation = 0
                self.check_point=False
                while self.totalTrainedModel < self.MAX_SOL:
                    self.evolve_generation()
                    self.check_point = False
                    print(f"Generation:{self.generation}, Best: {self.best_arch.fitness}, {self.best_arch.solNo}")
                    self.generation = 1 + self.generation
                    self.log_dic['generation'] = self.generation
                    with open('../log/check_point_idrid.txt', 'w') as f:
                        f.write(str(self.log_dic))
            # 如果不是在初始化种群中断，则继续进化
            else:
                self.reset()
                f = open('../log/check_point_idrid.txt', 'r')
                self.log_dic = ast.literal_eval(f.read())
                self.best_model_numb = self.log_dic['best_model_numb']
                self.best_arch = readPickleFile(self.best_model_numb)
                self.pop_size = pop_size
                self.allModels = self.log_dic['allModels']
                self.generation = self.log_dic['generation']
                self.solNo = pop_size+pop_size*(self.generation+1)
                self.totalTrainedModel = self.log_dic['solNo']
                self.pop_solNo = self.log_dic['pop_solNo']
                for i in self.pop_solNo:
                    temp = readPickleFile(i)
                    self.population.append(temp)
                self.population = np.array(self.population)
                while self.totalTrainedModel < self.MAX_SOL:
                    self.evolve_generation()
                    self.check_point=False
                    print(f"Generation:{self.generation}, Best: {self.best_arch.fitness}, {self.best_arch.solNo}")
                    self.generation = 1 + self.generation
                    self.log_dic['generation'] = self.generation
                    with open('../log/check_point_idrid.txt', 'w') as f:
                        f.write(str(self.log_dic))
        elif self.check_point and len(os.listdir('H:\\hmx\\NAS_data\\idrid\\models_de'))==0:
            self.seed = seed
            self.solNo = 0
            self.generation = 0
            self.totalTrainedModel = 0
            print(self.mutation_strategy)
            self.reset()
            # 获取种群初始适应度
            self.init_eval_pop()
            # 训练模型总数self.MAX_SOL=500
            while self.totalTrainedModel < self.MAX_SOL:
                self.evolve_generation()
                self.check_point=False
                print(f"Generation:{self.generation}, Best: {self.best_arch.fitness}, {self.best_arch.solNo}")
                self.generation = 1 + self.generation
                self.log_dic['generation'] = self.generation
                with open('../log/check_point_idrid.txt', 'w') as f:
                    f.write(str(self.log_dic))