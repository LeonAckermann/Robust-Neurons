import json
import os
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET


#from tools.dataset_tool import dfs_search


class restaurantDataset(Dataset):
    def __init__(self, config, mode,encoding="utf8", *args, **params):
        #self.config = config
        #self.mode = mode
        #self.data_path = config.get("data", "%s_data_path" % mode)
        #self.data_path = config.get("data", "train_data_path")
        #data = json.load(open(self.data_path, "r"))
        if config.get("eval", "compute_baseline") == "True":
            self.data_path = "./data/Restaurants_Train.xml"
        else:
            self.data_path = "./data/Restaurants_Test_Gold.xml"
        # Load and parse XML data
        tree = ET.parse(self.data_path)
        root = tree.getroot()

        self.data = []
        #nones = 0
        emo_dict = {"positive": 2, "neutral": 1, "negative": 0, "conflict": 3}

        for sentence in root.findall("sentence"):
            text = sentence.find("text").text.strip()
            aspect_terms = sentence.findall("aspectTerms/aspectTerm")
            #print(aspect_terms)
            if aspect_terms is not None:
                for aspect_term in aspect_terms:
                    #print(aspect_term)
                    term = aspect_term.attrib["term"].strip()
                    polarity = emo_dict[aspect_term.attrib["polarity"].strip()]
                    sent_label = {"sent": text + " " + term, "label": polarity}
                    self.data.append(sent_label)
            #else:
            #    nones += 1
            

        print("number of data", len(self.data) )
        #print(self.data[0])
        #print("nones", nones)
        #emo_dict={"positive":2,"neutral":1,"negative":0,"conflict":3}
        #emo_dict={"positive":0,"neutral":1,"negative":2}



    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
