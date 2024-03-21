import json
import os
from torch.utils.data import Dataset
import csv
#from .save_length import update_length
class ethicsdeontologyDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        #fin = csv.reader(open(self.data_path, "r"), delimiter=",")
        if config.get("eval", "debugging") == "True":
            path = "/Users/leonackermann/Desktop/continuous_prompt_analysis/skill_neurons/data/"
        else:
            path = "./data/"

        if config.get("eval", "compute_baseline") == "True":
            fin = csv.reader(open(f"{path}deontology_train.csv", "r"), delimiter=",")
        else:
            fin = csv.reader(open(f"{path}deontology_test.csv", "r"), delimiter=",")
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
                    self.data.append({"sent1": ins[1].strip(), "sent2": ins[2].strip(), "label": int(ins[0])})
            
            #file_path = f'./samples_used/{config.get("output", "model_name")}.json'
            #if current_part == 1:
            #    num_samples_dict = {"num_samples": len(self.data)}
            #    try:
            #        os.mkdir("./samples_used")
            #    except:
            #        pass
            #    with open(file_path, "w") as json_file:
            #        json.dump(num_samples_dict, json_file)
            #if current_part <= total_parts and current_part>1:
            #    num_samples = json.load(open(file_path, "r"))["num_samples"]
            #    num_samples += len(self.data)
            #    num_samples_dict = {"num_samples": num_samples}
            #    with open(file_path, "w") as json_file:
            #        json.dump(num_samples_dict, json_file)
            update_length(config, len(self.data))

        
        else:
            self.data = [{"sent1": ins[1].strip(), "sent2": ins[2].strip(), "label": int(ins[0])} for ins in fin]
        print("the number of data", len(self.data))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
