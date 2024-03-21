import json

def update_length(config, length):
    file_path = f'./samples_used/{config.get("output", "model_name")}.json'
    current_part = config.getint("eval", "current_part")
    total_parts = config.getint("eval", "total_parts")-1
    if current_part == 1:
        num_samples_dict = {"num_samples": length}
        try:
            os.mkdir("./samples_used")
        except:
            pass
        with open(file_path, "w") as json_file:
            json.dump(num_samples_dict, json_file)
    if current_part <= total_parts and current_part>1:
        num_samples = json.load(open(file_path, "r"))["num_samples"]
        num_samples += length
        num_samples_dict = {"num_samples": num_samples}
        with open(file_path, "w") as json_file:
            json.dump(num_samples_dict, json_file)