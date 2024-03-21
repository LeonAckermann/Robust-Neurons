import json
import os
from torch.utils.data import Dataset
import csv
from datasets import load_dataset
from .save_length import update_length


class IMDBDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        #self.config = config
        self.mode = mode
        ##self.data_path = config.get("data", "%s_data_path" % mode)
        #self.encoding = encoding

        #fin = csv.reader(open(self.data_path, "r"), delimiter="\t", quotechar='"')

        fin = load_dataset('imdb')
        #print(fin)
        if config.get("eval", "compute_baseline") == "True":
            fin = [row for row in fin['train']]
        else:
            fin = [row for row in fin['test']]
        label_map = {"positive":1, "negative":0}

        

        if config.get("eval", "compute_in_parts") == "True" and config.get("eval", "compute_baseline") == "True":

            current_part = config.getint("eval", "current_part")
            total_parts = config.getint("eval", "total_parts")-1
            print("dataset part", current_part)
            print("total parts", total_parts)
            lower_bound = int((current_part-1)*len(fin)/total_parts)
            upper_bound = int(current_part*len(fin)/total_parts)

            self.data = []
            for idx, ins in enumerate(fin):
                if idx > lower_bound and idx <= upper_bound:
                    self.data.append({"sent": ins['text'].strip(), "label":ins['label']})

            update_length(config, len(self.data))

        elif mode == "test":
            self.data = [{"sent": ins[0].strip()} for ins in fin]
        else:
            self.data = [{"sent": ins['text'].strip(), "label":ins['label']} for ins in fin]
            #print(self.data)
            #exit()
        print(self.mode, "the number of data", len(self.data))
       
        # from IPython import embed; embed()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
    
