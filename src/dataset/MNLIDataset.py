import json
import os
from torch.utils.data import Dataset
import csv
from datasets import load_dataset
from .save_length import update_length


class MNLIDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.data = load_dataset('glue', 'mnli')
        self.train_data = self.data['train']
        #self.mismatched_data = self.data['train_mismatched']
        self.matched_data = self.data['validation_matched']
        self.mismatched_data = self.data['validation_mismatched']
        self.mode = mode

        #org_dict = {"contradiction":2,"neutral":1,"entailment":0}
        #after_dict = {"contradiction":0,"neutral":1,"entailment":2}
        #_dict = {2:0,1:1,0:2}
        _dict = {2:0, 0:1} # without neutral
        # now 0:contradiction, 1:entailment

        #print("bound", lower_bound, upper_bound)
        if config.get("eval", "compute_in_parts") == "True" and config.get("eval", "compute_baseline") == "True":

            current_part = config.getint("eval", "current_part")
            total_parts = config.getint("eval", "total_parts")-1
            print("dataset part", current_part)
            print("total parts", total_parts)
            lower_bound = int((current_part-1)*self.train_data.num_rows/total_parts)
            upper_bound = int(current_part*self.train_data.num_rows/total_parts)

            self.data = []

            for idx, ins in enumerate(self.train_data):
                if idx > lower_bound and idx <= upper_bound and ins["label"] != 1:
                    self.data.append({"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": _dict[ins['label']]})

            update_length(config, len(self.data))  
        elif config.get("eval", "compute_baseline") == "True" and config.get("eval", "compute_in_parts") == "False":    
            self.data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": _dict[ins['label']]} for idx, ins in enumerate(self.train_data) if ins["label"] != 1]
        else:
            if mode == "valid_matched":
                self.data = [{"sent1": ins['hypothesis'].strip(), 
                              "sent2": ins['premise'].strip(), 
                              "label": _dict[ins['label']]} for ins in self.matched_data if ins["label"] != 1]

            elif mode == "valid_mismatched":
                self.data = [{"sent1": ins['hypothesis'].strip(), 
                              "sent2": ins['premise'].strip(), 
                              "label": _dict[ins['label']]} for ins in self.mismatched_data if ins["label"] != 1]
            
        #else:
        #    print("first row of data", self.train_data[0])
        #    self.data = [{"sent1": ins['hypothesis'].strip(), 
        #                  "sent2": ins['premise'].strip(), 
        #                  "label": _dict[ins['label']]} for ins in self.train_data if ins["label"] != 1]
        
        print(self.mode, "the number of data", len(self.data))
        # from IPython import embed; embed()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
    