import json
import os
from torch.utils.data import Dataset
import csv
from datasets import load_dataset
import numpy as np

class AdvQNLIDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        #self.data = load_dataset('glue', 'qnli')
        if config.get("model", "official_submission") == "True":
            self.data_path = config.get("data", "data_path")
            #self.data_path = "./data/adv_glue.json"
        else:
            self.data_path = "./data/adv_glue.json"
        data = json.load(open(self.data_path, "r"))
        self.data = data["qnli"]
        #print(data)
        #self.train_data = self.data['train']
        #self.validation_data = self.data['validation']
        #self.test_data = self.data['test']

        #part = config.getint("eval", "part")
        #lower_bound = int((part-1)*self.train_data.num_rows/2)
        #upper_bound = int(part*self.train_data.num_rows/2)
        #print("train data rows", self.train_data.num_rows)
        #print(f"current part {lower_bound} - {upper_bound}")

        #ORG: 1: False, 0:True

        _map={0:1,1:0}
        #Now: 0: False, 1:True

        #if mode == "test":
        #    self.data = [{"sent1": ins['question'].strip(), "sent2": ins['sentence']} for ins in self.test_data]
        if mode == 'valid':
            self.data = [{"sent1": ins['question'].strip(), "sent2": ins['sentence'].strip(), "label": _map[ins['label']]} for ins in self.data]
        
        #if config.get("eval", "compute_baseline") == "True":
        ##if config == "True":
        #    self.data = []
        #    for idx, ins in enumerate(self.train_data):
        #        if idx > lower_bound and idx <= upper_bound:
        #            self.data.append({"sent1": ins['question'].strip(), "sent2": ins['sentence'].strip(), "label": _map[ins['label']]})
        #            #self.data = [{"sent1": ins['question'].strip(), "sent2": ins['sentence'].strip(), "label": _map[ins['label']]} for ins in self.train_data]
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
    adv_qnli = AdvQNLIDataset("False", "valid")
    var = adv_qnli.get_variance()
    print(var)
    


