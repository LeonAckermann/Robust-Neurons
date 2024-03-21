from dataset import dataset_list#
from torch.utils.data import DataLoader
#import formatter as form
from datasets import load_dataset
from models import get_model
import find


import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
#from tensorboardX import SummaryWriter
from timeit import default_timer as timer
import formatter as form
import json
import os
import numpy as np
import copy
#import cupy as cp
import itertools


from models.PromptT5 import PromptT5

from transformers import AutoConfig



# save formatting function to pass it to the dataloader
formatter = {}
global_config = []

def init_formatter(config):
    global_config.append(config)
    formatter['valid'] = form.init_formatter(config=config,mode='valid')
    
    def valid_collate_fn(data):
        return formatter['valid'].process(data, global_config, 'valid')
    
    return valid_collate_fn

def init_valid_dataset(dataset_name, config, mode, batch_size, shuffle, reader_num, drop_last):

    if dataset_name in dataset_list.keys():
        if mode in ["valid", "test"] and "MNLI" in dataset_name:
            if config.get("data", "matched"):
                mode = mode + "_matched"
            else:
                mode = mode + "_mismatched"

        dataset = dataset_list[dataset_name](config=config, mode=mode)
        collate_fn = init_formatter(config)
        #print(collate_fn)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=reader_num,
                                collate_fn=collate_fn,
                                drop_last=drop_last)
        return dataloader
    
def init_dataset(dataset_name, config, mode, batch_size, shuffle, reader_num, drop_last):
    
    init_formatter(config)
    adversarial = config.get("data", "adversarial")
    print("adversarial", adversarial, type(adversarial))
    if config.get("data", "adversarial") == "True":
        return init_valid_dataset(f"Adv{dataset_name}", config, mode, batch_size, shuffle, reader_num, drop_last)
    else:
        return init_valid_dataset(dataset_name, config, mode, batch_size, shuffle, reader_num, drop_last)

def init_prompt(config):
    if config.get("eval", "prompt_dataset") == None:
        prompt_name = config.get("output", "model_name")
    else:
        prompt_name = f"{config.get('eval', 'prompt_dataset')}Prompt{config.get('model', 'model_base')}"
    seed = config.get('eval', 'prompt_seed')
    if config.get("eval", "debugging") == "True":
        load_task_prompt_dir = "/Users/leonackermann/Desktop/continuous_prompt_analysis/continuous_prompts/"+f"seed_{seed}/"+prompt_name+"/task_prompt"
    else:
        load_task_prompt_dir = "../continuous_prompts/"+f"seed_{seed}/"+prompt_name+"/task_prompt" 
    prompt_emb = torch.load(load_task_prompt_dir, map_location=torch.device('cpu')) # force to load on cpu
    return prompt_emb

def init_model(config, prompt_emb, gpu_list):
    
    model = get_model(config.get("model", "model_name"))(size=config.get("model", "model_size"), config=config)
    ##Put prompt emb back to model
    if config.get("model", "model_base") == "Roberta":
        model.encoder.roberta.embeddings.prompt_embeddings.weight.data = prompt_emb
    elif config.get("model", "model_base") == "Bert":
        model.encoder.bert.embeddings.prompt_embeddings.weight.data = prompt_emb
    elif config.get("model", "model_base") == "T5":
        #model.encoder.t5.embeddings.prompt_embeddings.weight.data = prompt_emb
        model.encoder.prompt_embeddings.weight.data = prompt_emb
        model.encoder.encoder.prompt_tokens.weight.data = prompt_emb
        model.encoder.decoder.prompt_tokens.weight.data = prompt_emb
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    #print(f"Model on {device}")
    return model

def init_all(config, gpu_list):
    
    prompt_emb = init_prompt(config)
    model = init_model(config, prompt_emb, gpu_list)
    dataset = init_dataset(dataset_name=config.get("data", "valid_dataset_type"), 
                           config=config, 
                           mode="valid", 
                           batch_size=config.getint("eval", "batch_size"), 
                           shuffle=False, 
                           reader_num=config.getint("eval", "reader_num"), 
                           drop_last=False)
    if config.get("eval", "perturbation_analysis") == "True":
        mask = init_mask(config)
    else:
        mask = None
    return dataset, model, mask

def init_perturbation_mask(config, model_config, mean, std, n, neurons, predictivities):
    
    """init mask with gaussian noise of mean and std for skill neurons (already sorted and cut to topn)

    args:
        - config: config file
        - model_config: model config file
        - mean: mean of gaussian noise
        - std: std of gaussian noise
        - n: number of neurons to perturb
        - skill_neurons: 
            - numpy array of shape (n,2) if config.getint("eval", "dim") == 2
            - numpy array of shape (n,3) if config.getint("eval", "dim") == 3

    returns:
        - mask
    
    """
    if config.get("model", "model_base") == "T5":
        mask = np.zeros((model_config.num_layers, 230, model_config.d_ff))
    else:
        mask = np.zeros((model_config.num_hidden_layers, 231, model_config.intermediate_size))
    #length = len(skill_neurons)
    noise = np.random.normal(mean, std, size=len(neurons))
    #print("noise", noise)
    #print("skill neurons", neurons)
    if config.get("eval", "mask") == "skill_neurons":
        if config.get("eval", "mode") == "max_mean_100":
            for i, neuron in enumerate(neurons):
                layer, index = neuron
                
                mask[layer,:100,index] = noise[i]
        if config.get("eval", "mode") == "max_mean_230":
            for i, neuron in enumerate(neurons):
                layer, index = neuron
                if config.get("eval", "perturbation_type") == "noise":
                    mask[layer,:,index] = noise[i]
                else:
                    mask[layer,:,index] = 1
        if config.get("eval", "mode") == "max_mean_token":
            for i, neuron in enumerate(neurons):
                layer, index = neuron
                token = np.argmax(predictivities[layer,:,index])
                mask[layer,token,index] = noise[i]
    elif config.get("eval", "mask") == "random_neurons":
        if config.get("eval", "mode") == "max_mean_100":
            for i, neuron in enumerate(neurons):
                layer, _, index = neuron
                mask[layer,:100,index] = noise[i]
        if config.get("eval", "mode") == "max_mean_230":
            for i, neuron in enumerate(neurons):
                layer, index = neuron
                perturbation_type = config.get("eval", "perturbation_type") 
                if config.get("eval", "perturbation_type") == "noise":
                    mask[layer,:,index] = noise[i]
                else:
                    mask[layer,:,index] = 1
        if config.get("eval", "mode") == "max_mean_token":
            for i, neuron in enumerate(neurons):
                layer,token,index = neuron
                mask[layer,token,index] = noise[i]
    #if predictivities != None: # this is if neurons are skill neurons
    #    if config.getint("eval", "dim") == 2:
    #        for i, neuron in enumerate(skill_neurons):
    #            layer, index = neuron
    #            token = np.argmax(predictivities[layer,:,index])
    #            mask[layer,token,index] = noise[i]
    #    else:
    #        for i, neuron in enumerate(skill_neurons):
    #            layer, token, index = neuron
    #            mask[layer,token,index] = noise[i]
    #else: # this is if neurons are random neurons
    #    if config.getint("eval", "dim") == 2:
    #        for i, neuron in enumerate(skill_neurons):
    #            layer, token, index = neuron
    #            mask[layer,token,index] = noise[i]
    #    else:
    #        for i, neuron in enumerate(skill_neurons):
    #            layer, token, index = neuron
    #            mask[layer,token,index] = noise[i]
    if config.get("eval", "perturbation_type") == "suppression":
        return mask.astype(bool)
    return mask

def init_random_perturbation_mask(config, model_config, mean, std, n):
    """not used anymore
    
    init random mask with gaussian noise of mean and std with n neurons perturbed
    
    args:
        - config: config file
        - model_config: model config file
        - mean: mean of gaussian noise
        - std: std of gaussian noise
        - n: number of neurons to perturb

    returns:
        - mask: numpy array of shape (num_layers, 230, num_neurons) if t5
        - mask: numpy array of shape (num_hidden_layers, 231, num_neurons) if roberta
    """

    if config.get("model", "model_base") == "T5":
        mask = np.zeros((model_config.num_layers, 230, model_config.d_ff))
        rand_layers = np.random.randint(0, model_config.num_layers, size=n)
        #rand_tokens = np.random.randint(0, 230, size=n)
        rand_indices = np.random.randint(0, model_config.d_ff, size=n)
        indices = np.stack([rand_layers, rand_indices], axis=1)
    else:
        mask = np.zeros((model_config.num_hidden_layers, 231, model_config.intermediate_size))
        rand_layers = np.random.randint(0, model_config.num_hidden_layers, size=n)
        #rand_tokens = np.random.randint(0, 231, size=n)
        rand_indices = np.random.randint(0, model_config.intermediate_size, size=n)
        indices = np.stack([rand_layers, rand_indices], axis=1)

    #print("random neurons", indices)
    noise = np.random.normal(mean, std, size=n)
    if config.getint("eval", "dim") == 2:
        for i, neuron in enumerate(indices):
            layer, index = neuron
            mask[layer,:,index] = noise[i]
    else:
        for i, neuron in enumerate(indices):
            layer, token, index = neuron
            mask[layer,token,index] = noise[i]
    return mask

def init_empty_mask(config, model_config):
    if config.get("model", "model_base") == "T5":
        mask = np.zeros((model_config.num_layers, 230, model_config.d_ff))
    else:
        mask = np.zeros((model_config.num_hidden_layers, 231, model_config.intermediate_size))

    return mask

def init_mask(config):
    """init mask for perturbation analysis based on config

    Args:
        - if config.get("eval", "mask")=="skill_neurons" then the mask is initialized with the skill neurons
        - if config.get("eval", "mask")=="random_neurons" then the mask is initialized with random neurons
    
    Returns:
        - mask: numpy array of shape (num_layers, 230, num_neurons) if t5 and config.getint("eval", "mask") != None
        - mask: numpy array of shape (num_hidden_layers, 231, num_neurons) if roberta and config.getint("eval", "mask") != None 
        - mask: None if config.getint("eval", "mask") == None
    """
    if config.get("model","model_base") == "T5":
        model_config = AutoConfig.from_pretrained(f"{config.get('model','model_base').lower()}-{config.get('model','model_size').lower()}")
    elif config.get("model", "model_base") == "Roberta":
        model_config = AutoConfig.from_pretrained(f"{config.get('model','model_base').lower()}-{config.get('model','model_size').lower()}")
    elif config.get("model", "model_base") == "Bert":
        model_config = AutoConfig.from_pretrained(f"{config.get('model','model_base').lower()}-{config.get('model','model_size').lower()}-uncased")

    #if config.getint("eval", "mode") == "max_mean_100":
    #    skill_neuron_path = "skill_neurons_across_model_dim_2"
    #    predictivities_path = "../../../../../../Volumes/Extreme SSD/Projects/20230501_bachelor_thesis/neuron_predictivities_raw/" # shape(12,100,3072) if roberta, shape(2,12,100,3072) if t5
    #else:
    #    skill_neuron_path = "skill_neurons_across_model_dim_3"
#
    #if config.getint("eval", "dim") == 2:
    #    random_neuron_path = "random_neurons_across_model_dim_2_with_token"
    #    #random_neuron_path = "random_neurons_across_model_dim_2"
    #else:
    #    random_neuron_path = "random_neurons_across_model_dim_3"

    random_neuron_path = "random_neurons_across_model_dim_2_with_token"
    if config.get("eval", "layer_model_mode") == "layer":
        skill_neuron_path = "skill_neuron_across_layer"
        random_neuron_path = "random_neuron_across_layer"
    else:
        skill_neuron_path = "skill_neurons_across_model_dim_2"
    #predictivities_path = "../../../../../../Volumes/Extreme SSD/Projects/20230501_bachelor_thesis/neuron_predictivities_raw/" # shape(12,100,3072) if roberta, shape(2,12,100,3072) if t5
    predictivities_path = "./neuron_predictivities"
    seed = config.get('eval', 'prompt_seed')
    if config.get("eval", "adv_nonAdv") == "False":
        if config.get("data", "adversarial") == "True":
            skill_neuron_name = f"Adv{config.get('output','model_name')}_mean.npy"
        else:
            skill_neuron_name = f"{config.get('output','model_name')}_mean.npy"
    elif config.get("eval", "adv_nonAdv") == "True":
        if config.get("data", "adversarial") == "True":
            skill_neuron_name = f"{config.get('output','model_name')}_mean.npy"
        else:
            skill_neuron_name = f"Adv{config.get('output','model_name')}_mean.npy"

    if config.get("eval", "debugging") == "True":
        skill_neurons = np.load(f"/Users/leonackermann/Desktop/continuous_prompt_analysis/skill_neurons/{skill_neuron_path}/{skill_neuron_name}")
    else:
        #skill_neurons = np.load(f"./skill_neurons/{skill_neuron_path}/{skill_neuron_name}")
        # print working directory
        skill_neurons = np.load(f"./{skill_neuron_path}/{skill_neuron_name}")

    #elif config.get("eval", "adv_non-adv") == "False":
    #    if config.get("data", "adversarial") == "True":
    #        skill_neurons = np.load(f"./{skill_neuron_path}/Adv{config.get('output','model_name')}_mean.npy")
    #    else:
    #        skill_neurons = np.load(f"./{skill_neuron_path}/{config.get('output','model_name')}_mean.npy")
    #elif config.get("eval", "adv_non-adv") == "True":
    #    if config.get("data", "adversarial") == "True":
    #        skill_neurons = np.load(f"./{skill_neuron_path}/{config.get('output','model_name')}_mean.npy")
    #    else:
    #        skill_neurons = np.load(f"./{skill_neuron_path}/Adv{config.get('output','model_name')}_mean.npy")
    

    if config.get("eval", "debugging") == "True":
        random_neurons = np.load(f"/Users/leonackermann/Desktop/continuous_prompt_analysis/skill_neurons/{random_neuron_path}/{config.get('output','model_name')}.npy")
    elif config.get("data", "adversarial") == "True":
        random_neurons = np.load(f"./{random_neuron_path}/Adv{config.get('output','model_name')}.npy")
    else:
        random_neurons = np.load(f"./{random_neuron_path}/{config.get('output','model_name')}.npy")

    if config.get("data", "adversarial") == "True":
        predictivities = np.load(f"{predictivities_path}/Adv{config.get('output','model_name')}_{seed}.npy")
    else:
        predictivities = np.load(f"{predictivities_path}/{config.get('output','model_name')}_{seed}.npy")

    if config.get("model","model_base") == "T5":
        predictivities = predictivities[0]
    
    if config.get("eval", "layer_model_mode") == "model":
        number_neurons = skill_neurons.shape[0]
    else:
        number_neurons = skill_neurons.shape[1]

    topn = float(config.get("eval", "topn"))
    #percent = int(number_neurons/100*config.getint("eval", "topn"))
    percent = int(number_neurons/100*topn)
    print("number of perturbed neurons", percent)
    if config.get("eval", "layer_model_mode") == "layer":
        skill_neurons_initial = skill_neurons[:,:percent]
        skill_neurons = skill_neurons_initial.reshape(-1,2)
        random_neurons_initial = random_neurons[:,:percent]
        random_neurons = random_neurons_initial.reshape(-1,2)
    else:
        skill_neurons = skill_neurons[:percent]
        random_neurons = random_neurons[:percent]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

  

    
    if config.get("eval", "mask")=="skill_neurons":
        print("skill neuron mask")
        mask = init_perturbation_mask(config, 
                                      model_config, 
                                      config.getfloat("eval", "mean"), 
                                      config.getfloat("eval", "std"), 
                                      topn,
                                      skill_neurons,
                                      predictivities)
        if config.get("eval", "perturbation_type") == "suppression":
            mask = torch.from_numpy(mask).bool().to(device)
        else:
            mask = torch.from_numpy(mask).float().to(device)
    elif config.get("eval", "mask")=="random_neurons":
        print("random neuron mask")
        #mask = init_random_perturbation_mask(config, 
        #                                     model_config, 
        #                                     config.getfloat("eval", "mean"), 
        #                                     config.getfloat("eval", "std"), 
        #                                     percent)
        mask = init_perturbation_mask(config, 
                                      model_config, 
                                      config.getfloat("eval", "mean"), 
                                      config.getfloat("eval", "std"), 
                                      topn,
                                      random_neurons,
                                      predictivities)
        if config.get("eval", "perturbation_type") == "suppression":
            mask = torch.from_numpy(mask).bool().to(device)
        else:
            mask = torch.from_numpy(mask).float().to(device)
    else:
        mask = None

    #print("mask", mask)
    return mask

def gen_time_str(t):
    t = int(t)
    minute = t // 60
    second = t % 60
    return '%2d:%02d' % (minute, second)

def output_value(epoch, mode, step, time, loss, info, end, config):
    try:
        delimiter = config.get("output", "delimiter")
    except Exception as e:
        delimiter = " "
    s = ""
    s = s + str(epoch) + " "
    while len(s) < 7:
        s += " "
    s = s + str(mode) + " "
    while len(s) < 14:
        s += " "
    s = s + str(step) + " "
    while len(s) < 25:
        s += " "
    s += str(time)
    while len(s) < 40:
        s += " "
    s += str(loss)
    while len(s) < 48:
        s += " "
    s += str(info)
    s = s.replace(" ", delimiter)
    if not (end is None):
        print(s, end=end)
    else:
        print(s)


def save_activation_states(activation_states):
    batch_activations_dict = {}
    
    
    for batch, batch_activations in activation_states.items():
        encoder_decoder_activations_dict = {}
        for encoder_decoder, encoder_decoder_activations in batch_activations.items():
            layer_activations_dict = {}
            for layer, layer_activations in encoder_decoder_activations.items():
                layer_activations_np = layer_activations.numpy().tolist()
                layer_activations_dict[layer] = layer_activations_np

            encoder_decoder_activations_dict[encoder_decoder] = layer_activations_dict
        batch_activations_dict[batch] = encoder_decoder_activations_dict

    with open("activation_states.json", "w") as file:
        json.dump(batch_activations_dict, file)


def valid(config, model, gpu_list, epoch, dataset_name, dataset, output_function, mask, mode="valid",  **kwargs):

    model.eval()

    acc_result = None
    total_loss = 0
    predict = []
    cnt = 0
    print("dataset", dataset)
    total_len = len(dataset)
    start_time = timer()
    output_info = ""
    activation_states_bsl = {}
    activation_states_bsl_list = []

    labels = []
    seed = config.get("eval", "prompt_seed")
   
    #print("model config", model_config)
    if config.get("model","model_base") == "T5":
        model_config = AutoConfig.from_pretrained(f"{config.get('model','model_base').lower()}-{config.get('model','model_size').lower()}")
        neuron_accuracy = np.zeros((2,model_config.num_layers,100,model_config.d_ff))
    elif config.get("model", "model_base") == "Roberta":
        model_config = AutoConfig.from_pretrained(f"{config.get('model','model_base').lower()}-{config.get('model','model_size').lower()}")
        neuron_accuracy = np.zeros((model_config.num_hidden_layers,100,model_config.intermediate_size))
    elif config.get("model", "model_base") == "Bert":
        model_config = AutoConfig.from_pretrained(f"{config.get('model','model_base').lower()}-{config.get('model','model_size').lower()}-uncased")
        neuron_accuracy = np.zeros((model_config.num_hidden_layers,100,model_config.intermediate_size))


    data_size = 0
    #local_rank = config.getint('distributed', 'local_rank')


    output_time = config.getint("output", "output_time")
    step = -1
    more = ""
    if total_len < 10000:
        more = "\t"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for step, data in enumerate(dataset):

            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if len(gpu_list) > 0:
                        data[key] = Variable(data[key].to(device))
                        #print(f"data key on cuda")
                    else:
                        data[key] = Variable(data[key])
                        print("data key on cpu")

            results = model(config, dataset_name, data, acc_result, "valid", mask, args=kwargs)
            #print("results", results)

            if config.get("eval", "compute_baseline") == "True":
                #print("shape of get function", model.get_activation_states()["encoder"]["layer 0"].shape)
                activation_states_bsl[f"batch {step}"] = find.a_bsl_to_cpu(copy.deepcopy(model.get_activation_states()), config)
                #activation_states_bsl_list.append(model.get_activation_states())
                #activation_states_bsl[f"batch {step}"] = find.one_batch_a_bsl(model.get_activation_states(), config, model_config)
                #activation_states_bsl_list.append(find.one_batch_a_bsl(model.get_activation_states_list(), config, model_config))

            if config.get("eval", "compute_predictivity") == "True" or config.get("eval", "compute_accuracy") == "True":

                activation_states_val = copy.deepcopy(model.get_activation_states())
                #print("activation states val keys", activation_states_val.keys())
                path = f"./baseline_activations/{config.get('output','model_name')}_{seed}.npy"
                baseline_activations = np.load(path)
                neuron_accuracy += find.compute_accuracy(baseline_activations, activation_states_val, data, config, model_config)
                #print(neuron_accuracy)
                #labels.append(data["label"])

            
            
            #print("results", results)
            if "T5" in config.get("model","model_base"):
                acc_result = results["acc_result"]
                total_loss += float(0)
            else:
                loss, acc_result = results["loss"], results["acc_result"]
                total_loss += float(loss)

            if config.get("model", "official_submission") == "True":
                predict.append(copy.deepcopy(results["predict"].numpy().tolist()))
            #print("predict", predict)
            cnt += 1
            

            data_size += data["label"].cpu().numpy().shape[0]

            
            delta_t = timer() - start_time
            output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                         "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)
            
            #running_acc = find.acc(activation_states_bsl, config)

    
    #print("länge datensatz", dataset.__len__())
    #print("data size", data_size)

    if config.get("model", "official_submission") == "True":
        predict = list(itertools.chain.from_iterable(predict))
        print("len predict", len(predict))
        #print("final predict", predict)
        predict_dict = {config.get("data", "valid_dataset_type").lower(): predict}
        try:
            os.mkdir("./predicts")
            print("Predicts directory created successfully.")
        except:
            print("Predicts directory already exists.")
        with open(f"./predicts/{config.get('output','model_name')}_{seed}.json", "w") as file:
            json.dump(predict_dict, file)
#
    if config.get("eval", "compute_predictivity") == "True":
        neuron_accuracy_temp = neuron_accuracy / data_size
        neuron_predictivity = np.where(neuron_accuracy_temp > 1 - neuron_accuracy_temp, neuron_accuracy_temp, 1 - neuron_accuracy_temp)
        try:
            os.mkdir("./neuron_predictivities")
            print("Predictivity directory created successfully.")
        except:
            print("Predictivity directory already exists.")
        if config.get("data", "adversarial") == "True":
            path = f"./neuron_predictivities/Adv{config.get('output','model_name')}_{seed}.npy"
        else:
            path = f"./neuron_predictivities/{config.get('output','model_name')}_{seed}.npy"
        np.save(path, neuron_predictivity)


    if config.get("eval", "compute_accuracy") == "True":
        neuron_accuracy = neuron_accuracy / data_size
        try:
            os.mkdir("./neuron_accuracy")
            print("Accuracy directory created successfully.")
        except:
            print("Accuracy directory already exists.")
        if config.get("data", "adversarial") == "True":
            path = f"./neuron_accuracy/Adv{config.get('output','model_name')}_{seed}.npy"
        else:
            path = f"./neuron_accuracy/{config.get('output','model_name')}_{seed}.npy"
        np.save(path, neuron_accuracy)
        
    if config.get("eval", "find_skill_neurons") == "True":
        input_path = f"./neuron_predictivities/{config.get('output','model_name')}.npy"
        neuron_predictivities = np.load(input_path)
        skill_neurons = find.skill_neurons(neuron_predictivities, config)
        try:
            os.mkdir("./skill_neurons")
            print("Skill neurons directory created successfully.")
        except:
            print("Skill neurons directory already exists.")
        #os.mkdir("./skill_neurons")
        if config.get("data", "adversarial") == "True":
            output_path = f"./skill_neurons/Adv{config.get('output','model_name')}_{seed}.npy"
        else:
            output_path = f"./skill_neurons/{config.get('output','model_name')}_{seed}.npy"
        np.save(output_path, skill_neurons)

    if config.get("eval", "compute_baseline") == "True":
        baseline_activations = find.a_bsl(activation_states_bsl, config, model_config)
        #baseline_activations = find.all_batches_a_bsl(activation_states_bsl, config, model_config)
        #baseline_activations = find.all_batches_a_vsl(activation_states_bsl_list, config, model_config)

        ########## use parts for QQP and QNLI ##########
        
        
        try:
            os.mkdir("./baseline_activations")
            print("Baseline directory created successfully.")
        except:
            print("Baseline directory already exists.")

        if config.get("eval", "compute_in_parts") == "True":
            try:
                os.mkdir(f"./baseline_activations/{config.get('output','model_name')}_{seed}")
            except:
                print("Model directory exists already.")
            print("save current part")
            path = f"./baseline_activations/{config.get('output','model_name')}_{seed}/{config.get('output','model_name')}_{seed}_{config.get('eval', 'current_part')}.npy"
        
        elif config.get("data", "valid_dataset_type") == "QQP" or config.get("data", "valid_dataset_type") == "QNLI":
            path = f"./baseline_activations/{config.get('output','model_name')}_{seed}_{config.get('eval', 'part')}.npy"
        else:
            path = f"./baseline_activations/{config.get('output','model_name')}_{seed}.npy"    

        
        #os.mkdir("./baseline_activations")
        #path = f"./baseline_activations/{config.get('output','model_name')}_{seed}.npy"
        np.save(path, baseline_activations)
    
    
    
    #if config.getboolean("distributed", "use"):
    #    shape = (len(acc_result), 4)
    #    # mytensor = torch.LongTensor([acc_result[key] for key in acc_result]).to(gpu_list[local_rank])
    #    print(acc_result)
    #    mytensor = torch.LongTensor([[int(key["TP"]), int(key["FN"]), int(key["FP"]), int(key["TN"])] for key in acc_result]).to(gpu_list[local_rank])
    #    mylist = [torch.LongTensor(shape[0], shape[1]).to(gpu_list[local_rank]) for i in range(config.getint('distributed', 'gpu_num'))]
    #    # print('shape', shape)
    #    # print('mytensor', mytensor.shape)
    #    torch.distributed.all_gather(mylist, mytensor)#, 0)
    #    if local_rank == 0:
    #        mytensor = sum(mylist)
    #        index = 0
    #        for i in range(len(acc_result)):
    #            acc_result[i]['TP'], acc_result[i]['FN'], acc_result[i]['FP'], acc_result[i]['TN'] = int(mytensor[i][0]), int(mytensor[i][1]), int(mytensor[i][2]), int(mytensor[i][3])
    #        # for key in acc_result:
    #        #     acc_result[key] = int(mytensor[index])
    #        #     index += 1
    #if local_rank <= 0:
    delta_t = timer() - start_time
    output_info = output_function(acc_result, config)

    output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
        gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                "%.3lf" % (total_loss / (step + 1)), output_info, None, config)
    
    #delta_t = timer() - start_time
    #output_info = output_function(acc_result, config)  
    #output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
    #    gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
    #            "%.3lf" % (total_loss / (step + 1)), output_info, None, config)
    
    try:
        #dataset_name = dataset_name
        #dataset_name = kwargs.checkpoint.split("/")[1]
        seed = config.get("eval", "prompt_seed")
        if config.get("eval", "adv_nonAdv") == "False":
            if config.get("data", "adversarial") == "True":
                dataset_name = f'Adv{config.get("output","model_name")}_{seed}'
                model_name = f'Adv{config.get("output","model_name")}'
            else:
                dataset_name = config.get("output","model_name")+"_"+seed
                model_name = config.get("output","model_name")
        elif config.get("eval", "adv_nonAdv") == "True":
            if config.get("data", "adversarial") == "True":
                dataset_name = f'Adv{config.get("output","model_name")}_{seed}_sn_{config.get("data", "valid_dataset_type")}'
                model_name = f'Adv{config.get("output","model_name")}'
            else:
                dataset_name = f'{config.get("output","model_name")}_{seed}_sn_{config.get("data", "valid_dataset_type")}'
                model_name = config.get("output","model_name")

        if config.get("eval", "perturbation_analysis") == "True":
            ##dir_save = f"./result_{config.get('eval', 'mask')}/run_{config.get('eval','run')}/"+str(dataset_name)+f"_{seed}_topn-{config.get('eval', 'topn')}"
            #dir_save = f"./result2_mask_{config.get('eval', 'mask')}_{config.get('eval','mode')}/"+str(dataset_name)+f"_topn-{config.get('eval', 'topn')}"
            if config.get("eval", "topn") == "0.1":
                topn = "0_1"
            elif config.get("eval", "topn") == "0.2":
                topn = "0_2"
            elif config.get("eval", "topn") == "0.3":
                topn = "0_3"
            elif config.get("eval", "topn") == "0.4":
                topn = "0_4"
            elif config.get("eval", "topn") == "0.5":
                topn = "0_5"
            elif config.get("eval", "topn") == "0.6":
                topn = "0_6"
            elif config.get("eval", "topn") == "0.7":
                topn = "0_7"
            elif config.get("eval", "topn") == "0.8":
                topn = "0_8"
            elif config.get("eval", "topn") == "0.9":
                topn = "0_9"
            else:
                topn = config.get("eval", "topn")
            if config.get("eval", "adv_nonAdv") == "False":
                dir_save = f"./result2_suppression_{config.get('eval', 'mask')}_{config.get('eval','mode')}/"+str(dataset_name)+f"_topn-{topn}"
            elif config.get("eval", "adv_nonAdv") == "True":
                #dir_save = f"/Users/leonackermann/Desktop/continuous_prompt_analysis/skill_neurons/result2_suppression_{config.get('eval', 'mask')}_{config.get('eval','mode')}/"+str(dataset_name)+f"_topn-{topn}"
                dir_save = f"./result2_suppression_{config.get('eval', 'mask')}_{config.get('eval','mode')}/"+str(dataset_name)+f"_topn-{topn}"

        elif config.get("eval", "prompt_dataset") == None:
            dir_save = f"./result/{str(dataset_name)}"
        else:

            dir_save = f"./result_transferability/{model_name}_{config.get('eval', 'prompt_dataset')}_{seed}"
        
        os.mkdir(dir_save)
        print("Path created:",dir_save)
    # exception for existing directory
    except:
        print("Path exists:",dir_save)

    with open(dir_save+"/"+"result.json", "w") as f:
        json.dump(output_info, f)

#if __name__ == '__main__':
#    break
    #dataset = init_dataset("squad", "no_mode", 32, True, 1, True)

    #model = PromptT5("base")
    #valid(model, "squad", dataset, None)
    #for i, batch in enumerate(dataset):
    #    print(i)
    #    print(batch)
    #    break