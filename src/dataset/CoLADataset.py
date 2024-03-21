import json
import os
from torch.utils.data import Dataset
import csv
from datasets import load_dataset

class CoLADataset(Dataset):
    def __init__(self, config, mode,encoding="utf8", *args, **params):
        #self.config = config
        #self.mode = mode
        self.data = load_dataset('glue', 'cola')
        if config.get("skill_neuron", "compute_baseline") == "True":
            data = self.data['train']
        else:
            data = self.data['validation']
        
        #self.data = [{"sent": ins['sentence'].strip(), "label": ins['label']} for ins in self.validation_data]
        self.data = [{"sent": ins['sentence'].strip(), "label": ins['label']} for ins in data]
     
        print( "the number of data", len(self.data))
        #print(self.data[0])
        # from IPython import embed; embed()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
    
