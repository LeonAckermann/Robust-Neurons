import json
import os
from torch.utils.data import Dataset
import csv
from datasets import load_dataset

class multi_newsDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        #self.data_path = config.get("data", "%s_data_path" % mode)
        self.encoding = encoding
        #self.data = json.load(open(self.data_path, "r"))
        if config.get("eval", "compute_baseline") == "True":
            self.data = load_dataset('multi_news', split='train')
        else:
            self.data = load_dataset('multi_news', split='validation')
        #if self.mode == "train":
        #    self.data = self.data[self.mode]
        #else:
        #    self.data = self.data["validation"]

        #print(self.data)
        data = [row for row in self.data]

        if mode == "test":
            #self.data = [{"sent": ins[0].strip()} for ins in data[1:]]
            #self.data = [{'context': ins.strip(), 'question': ins.strip()} for ins in data]
            #self.data = [{'context': ins["context"].strip(), 'question': ins["question"].strip()} for ins in data]
            self.data = [{'context': ins["document"].strip()} for ins in data]
        else:
            self.data = [{'context': ins["document"].strip(), 'label':ins['summary'].strip()} for ins in data]
        print(self.mode, "the number of data", len(self.data))
        # from IPython import embed; embed()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
    