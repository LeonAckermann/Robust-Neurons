import json
import os
from torch.utils.data import Dataset
import csv
from datasets import load_dataset

class squadDataset(Dataset):
    def __init__(self, config, mode):
        self.data = load_dataset('squad')
        if config.get("eval", "compute_baseline") == "True":
            self.data = self.data["train"]
        else:
            self.data = self.data["validation"]
        data = [row for row in self.data]
        self.data = [{'context': ins["context"].strip(), 
                      'question': ins["question"].strip(), 
                      'label':ins['answers']['text'][0].strip()} for ins in data]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
    
