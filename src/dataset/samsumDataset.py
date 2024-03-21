import json
import os
from torch.utils.data import Dataset
import csv
from datasets import load_dataset

class samsumDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        #self.data_path = config.get("data", "%s_data_path" % mode)
        #self.encoding = encoding
        #fin = csv.reader(open(self.data_path, "r"), delimiter="\t", quotechar='"')
        self.data = load_dataset('samsum')
        #print(self.data)
        #exit()
        if config.get("eval", "compute_baseline") == "True":
            self.data = self.data["train"]
        else:
            self.data = self.data["validation"]


        data = [row for row in self.data]

        if mode == "test":
            #self.data = [{"sent": ins[0].strip()} for ins in data[1:]]
            #self.data = [{'context': ins.strip(), 'question': ins.strip()} for ins in data]
            self.data = [{'context': ins["dialogue"].strip()} for ins in data]
        else:
            self.data = [{'context': ins["dialogue"].strip(), 'label':ins['summary'].strip()} for ins in data]
        print(self.mode, "the number of data", len(self.data))
        # from IPython import embed; embed()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
