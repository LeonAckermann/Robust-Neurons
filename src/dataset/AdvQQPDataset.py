import json
import os
from torch.utils.data import Dataset
import csv
from datasets import load_dataset
import numpy as np

class AdvQQPDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        #self.data = load_dataset('glue', 'qqp')
        if config.get("model", "official_submission") == "True":
            self.data_path = config.get("data", "data_path")
            #self.data_path = "./data/adv_glue.json"
        else:
            self.data_path = "./data/adv_glue.json"
        data = json.load(open(self.data_path, "r"))
        self.data = data["qqp"]
        #self.data = load_dataset('../data/')
        #self.train_data = self.data['train']
        #self.validation_data = self.data['validation']
        #self.test_data = self.data['test']
        #print(self.train_data.num_rows)
        #part = config.getint("eval", "part")
        #lower_bound = int((part-1)*self.train_data.num_rows/5)
        #upper_bound = int(part*self.train_data.num_rows/5)
        #0: false, 1:true

        #print(lower_bound, upper_bound)
        #if config.get("eval", "compute_baseline") == "True":
        #if config == "True":
        #    self.data = []
#
        #    for idx, ins in enumerate(self.train_data):
        #        if idx > lower_bound and idx <= upper_bound:
        #            self.data.append({"sent1": ins['question1'].strip(), "sent2": ins['question2'].strip(), "label": ins['label']})
        #    #self.data = [{"sent1": ins['question1'].strip(), "sent2": ins['question2'].strip(), "label": ins['label']} for idx, ins in enumerate(self.train_data)]
        #else:
        self.data = [{"sent1": ins['question1'].strip(), "sent2": ins['question2'].strip(), "label": ins['label']} for ins in self.data]
        
        print("first example", self.data[0])
        print(self.mode, "the number of data", len(self.data))
        # from IPython import embed; embed()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
    
    def get_variance(self):
        class_values = [ins['label'] for ins in self.data]
        var = np.array(class_values).var()
        return var
    
if __name__ == "__main__":
    qqp = AdvQQPDataset("True", "train")
    var = qqp.get_variance()
    print(var)
