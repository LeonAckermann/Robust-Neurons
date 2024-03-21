import random
import json
import os
import argparse

def create_random_results(dataset):
    data_path = "./data/adv_glue.json"
    data = json.load(open(data_path, "r"))
    data = data[dataset]
    length = len(data)
    random_preds = [random.choice([0, 1]) for _ in range(length)]
    random_preds_dict = {dataset: random_preds}
    with open(f"./predicts/{dataset}.json", "w") as file:
        json.dump(random_preds_dict, file)

def combine_all_dicts(datasets_eval, datasets_random, seed, model):
    # List of input JSON files
    #print(datasets)
    input_files_eval = [f"./predicts/{dataset}Prompt{model}_{seed}.json" for dataset in datasets_eval]
    input_files_random = [f"./predicts/{dataset}.json" for dataset in datasets_random]
    input_files = input_files_eval + input_files_random
    #print(input_files)
    # Read data from each file and store them in a list of dictionaries
    list_of_dicts = []
    for file in input_files:
        if os.path.exists(file):
            data_dict = read_json_file(file)
            list_of_dicts.append(data_dict)
    #print("list of dicts", list_of_dicts)
    # Combine all dictionaries into a single dictionary
    combined_dict = combine_dicts(list_of_dicts)
    #print(combined_dict)
    # Write the combined dictionary to a new JSON file
    output_file = f'./predicts/preds_{seed}.json'
    write_json_file(combined_dict, output_file)

def combine_dicts(list_of_dicts):
    combined_dict = {}
    for data_dict in list_of_dicts:
        for key in data_dict:
            combined_dict[key] = data_dict[key]
    return combined_dict

def write_json_file(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)


def read_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--seed')
    args = parser.parse_args()
    datasets_random = ["mnli", "mnli-mm", "rte"]
    for dataset in datasets_random:
        create_random_results(dataset)
    datasets_eval = ["SST2", "QQP", "QNLI"]
    combine_all_dicts(datasets_eval, datasets_random, args.seed, args.model)
