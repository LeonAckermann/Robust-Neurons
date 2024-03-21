import json
import os
from torch.utils.data import Dataset

#from tools.dataset_tool import dfs_search
from datasets import load_dataset

class tweetevalsentimentDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        #self.data_path = config.get("data", "%s_data_path" % mode)
        #self.data_path = config.get("data", "train_data_path")
        #data = json.load(open(self.data_path, "r"))
        # load tweeteval sentiment dataset from huggingface
        if config.get("eval", "compute_baseline") == "True":
            data = load_dataset('tweet_eval', 'sentiment', split='train')
        else:
            data = load_dataset('tweet_eval', 'sentiment', split='validation')
        print(data)
        '''
        self.data = []
        for rel in data:
            if mode == "train":
                inses = data[rel][:int(len(data[rel]) * 0.8)]
            else:
                inses = data[rel][int(len(data[rel]) * 0.8):]
            for ins in inses:
                ins["label"] = rel
                self.data.append(ins)
        '''

        '''
        for line in data:
            print(line)
        print("===")
        print(len(data))
        exit()
        '''

        emo_dict={"positive":2,"neutral":1,"negative":0,}
        #emo_dict={"positive":0,"neutral":1,"negative":2}

        if mode == "test":
            self.data = [{"sent": ins['sentence'].strip()} for ins in data]
        elif mode == 'valid':
            self.data = [{"sent": ins['text'].strip(), "label": ins['label']} for ins in data]
        else:
            self.data = [{"sent": ins['sentence'].strip(), "label": emo_dict[ins['label']]} for ins in data]
        print(self.mode, "the number of data", len(self.data))



    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
