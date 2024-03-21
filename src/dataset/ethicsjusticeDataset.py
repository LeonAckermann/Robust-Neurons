import json
import os
from torch.utils.data import Dataset
import csv
from .save_length import update_length


class ethicsjusticeDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        #fin = csv.reader(open(self.data_path, "r"), delimiter=",")
        if config.get("eval", "compute_baseline") == "True":
            fin = csv.reader(open("./data/justice_train.csv", "r"), delimiter=",")
        else:
            fin = csv.reader(open("./data/justice_test.csv", "r"), delimiter=",")
        fin = [row for row in fin if row[0]=='1' or row[0]=='0']


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
                    self.data.append({"sent": ins[1].strip(), "label": int(ins[0].strip())})

            update_length(config, len(self.data))
        else:
            self.data = [{"sent": ins[1].strip(), "label": int(ins[0].strip())} for ins in fin]
        print("the number of data", len(self.data))
        print(self.data[0])

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

# main function
if __name__ == "__main__":
    dataset = ethicsjusticeDataset()
    