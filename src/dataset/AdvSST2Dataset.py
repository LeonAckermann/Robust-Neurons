import json
import os
from torch.utils.data import Dataset
import csv
from datasets import load_dataset
import numpy as np

class AdvSST2Dataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        official_submission = config.get("model", "official_submission")
        print("official submission", official_submission, type(official_submission))
        if config.get("model", "official_submission") == "True":
            self.data_path = config.get("data", "data_path")
            #self.data_path = "./data/adv_glue.json"
        else:
            #self.data_path = "/Users/leonackermann/Desktop/continuous_prompt_analysis/skill_neurons/data/adv_glue.json"
            self.data_path = "./data/adv_glue.json"
        data = json.load(open(self.data_path, "r"))
        fin = data["sst2"]
        #print(data)
        #self.encoding = encoding
        ##fin = csv.reader(open(self.data_path, "r"), delimiter="\t", quotechar='"')
        #if config.get("eval", "compute_baseline") == "True":
        #    fin = load_dataset('sst2', split='train')
        #else:
        #    fin = load_dataset('sst2', split='validation')
        print("len data before", len(fin))
        data = [row for row in fin]
        print("len data after", len(data))
        print("data", data[0])
        #if "T5" in self.config.get("model","model_base"):
        if mode == "test":
            if mode == "test":
                self.data = [{"sent": ins[0].strip()} for ins in data[1:]]
        else:
            #self.data = [{"sent": ins[0].strip(), "label": int(ins[1].strip())} for ins in data[1:]]
            self.data = [{"sent": ins['sentence'].strip(), "label": int(ins['label'])} for ins in data]
        ##else:
        #if mode == "test":
        #    self.data = [{"sent": ins[0].strip()} for ins in data[1:]]
        #else:
        #    self.data = [{"sent": ins['sentence'].strip(), "label": int(ins['label'])} for ins in data[1:]]
        print("first example", self.data[0])
        print(self.mode, "the number of data", len(self.data))
        ## from IPython import embed; embed()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
    
    def get_variance(self):
        class_values = [ins['label'] for ins in self.data]
        var = np.array(class_values).var()
        return var
    
    
if __name__ == "__main__":
    adv_sst2 = AdvSST2Dataset("False", "train")
    var = adv_sst2.get_variance()
    print(var)

