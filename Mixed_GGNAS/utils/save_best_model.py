import datetime
import pickle

import torch


def writePickle(data, name,seed,count):
    # Write History
    with open("C:\\humengxiang\\DE_github\\other_models\\results\\model_{}_seed_{}_{}.pkl".format(name,seed,count), "wb") as pkl:
        pickle.dump(data, pkl)
class BestModelCheckPoint:
    def __init__(self, model_name,count):
        self.best_score = 0
        self.model_name = model_name
        self.count = count
    
    def check(self, score, model, seed):
        if score > self.best_score:
            print("Best Score:", score)
            self.best_score = score
            writePickle(model,self.model_name,seed,self.count)
            #torch.save(model.state_dict(), f"""{path}model_{self.model_name}_seed_{seed}_{self.count}.pt""")
