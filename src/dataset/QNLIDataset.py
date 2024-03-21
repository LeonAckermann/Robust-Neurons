import json
import os
from torch.utils.data import Dataset
import csv
from datasets import load_dataset
import numpy as np
from .save_length import update_length


class QNLIDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.data = load_dataset('glue', 'qnli')
        self.train_data = self.data['train']
        self.validation_data = self.data['validation']
        self.test_data = self.data['test']

        #part = config.getint("eval", "part")
        #lower_bound = int((part-1)*self.train_data.num_rows/2)
        #upper_bound = int(part*self.train_data.num_rows/2)
        #print("train data rows", self.train_data.num_rows)
        #print(f"current part {lower_bound} - {upper_bound}")

        #ORG: 1: False, 0:True

        _map={0:1,1:0}
        #Now: 0: False, 1:True

        if mode == "test":
            self.data = [{"sent1": ins['question'].strip(), "sent2": ins['sentence']} for ins in self.test_data]
        elif mode == 'valid':
            self.data = [{"sent1": ins['question'].strip(), "sent2": ins['sentence'].strip(), "label": _map[ins['label']]} for ins in self.validation_data]
        
        if config.get("eval", "compute_in_parts") == "True" and config.get("eval", "compute_baseline") == "True":

            current_part = config.getint("eval", "current_part")
            total_parts = config.getint("eval", "total_parts")-1
            print("dataset part", current_part)
            print("total parts", total_parts)
            lower_bound = int((current_part-1)*self.train_data.num_rows/total_parts)
            upper_bound = int(current_part*self.train_data.num_rows/total_parts)
        #if True == False:
        #if config == "True":
            self.data = []
            for idx, ins in enumerate(self.train_data):
                if idx > lower_bound and idx <= upper_bound:
                    self.data.append({"sent1": ins['question'].strip(), "sent2": ins['sentence'].strip(), "label": _map[ins['label']]})
        
            update_length(config, len(self.data))
        elif config.get("eval", "compute_in_parts") == "True":
            self.data = [{"sent1": ins['question'].strip(), "sent2": ins['sentence'].strip(), "label": _map[ins['label']]} for ins in self.train_data]
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
    adv_qnli = QNLIDataset("False", "valid")
    var = adv_qnli.get_variance()
    print(var)

