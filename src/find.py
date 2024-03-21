import numpy as np
#import cupy as cp
import tqdm
import csv
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import copy
import json
import copy
import argparse

import os

def a_bsl_to_cpu(activation_states_gpu, config):
    """put baseline activations to cpu
    
    args:
        activation_states_gpu (dict): 
            - dictionary with baseline activations on gpu 
            - if model is t5 dict has keys (encoder/decoder, layer) else only (layer)
        config (dict): dictionary with config parameters of parsed config file

    returns:
        activation_states_cpu (dict):
            - dictionary with baseline activations on cpu
            - if model is t5 dict has keys (encoder/decoder, layer) else only (layer)
    """
    if config.get("model","model_base") == "T5" : # if model is T5
        activation_states_cpu = {}
        for encoder_decoder_idx, encoder_decoder_key in enumerate(activation_states_gpu.keys()): # iterate over encoder and decoder
            activation_states_cpu[encoder_decoder_key] = {}
            for layer_idx, layer_key in enumerate(activation_states_gpu[encoder_decoder_key].keys()): # iterate over all layers
                activation_states_cpu[encoder_decoder_key][layer_key] = copy.deepcopy(activation_states_gpu[encoder_decoder_key][layer_key].cpu())
        return activation_states_cpu
    else:
        activation_states_cpu = {}
        for layer_idx, layer_key in enumerate(activation_states_gpu.keys()):
            activation_states_cpu[layer_key] = copy.deepcopy(activation_states_gpu[layer_key].cpu())
        return activation_states_cpu


#############################################################################################################
# reviewed #
#############################################################################################################
def a_bsl(activation_states, config, model_config):
    """Compute baseline activation for bert, roberta and t5, or save sum of baseline activations to file if compute_in_parts is True.

    Args:
        activation_states (dict): dictionary with activations of all batches
        config (dict): dictionary with config parameters of parsed config file
        model_config (dict): dictionary with config parameters of parsed model config file

    Returns:
        numpy array: array with shape (2, num_layers, seq_len, hidden_size) if t5 else array with shape (num_layers, seq_len, hidden_size)
    """
    if config.get("model","model_base") == "T5" : # if model is T5
        # create a numpy array of shape (2,num_layers,num_batches, seq_len, hidden_size)
        num_samples = 0

        batches_inside_layers_stacked = np.zeros((2, 
                                                  model_config.num_layers,
                                                  int(np.array(activation_states["batch 0"]["encoder"]["layer 0"][0].cpu()).shape[0]),
                                                  model_config.d_ff))
        
        for batch_idx, batch_key in tqdm.tqdm(enumerate(activation_states.keys()), total=len(activation_states.keys())): # iterate over all batches
            num_samples += np.array(activation_states[batch_key]["encoder"]["layer 0"].cpu()).shape[0]
            for encoder_decoder_idx, encoder_decoder_key in enumerate(activation_states[batch_key].keys()): # iterate over encoder and decoder
                for layer_idx, layer_key in enumerate(activation_states[batch_key][encoder_decoder_key].keys()): # iterate over all layers
                    #if np.array(activation_states[batch_key][encoder_decoder_key][layer_key].cpu()).shape[1] == 230: # if encoder
                    if encoder_decoder_key == "encoder":
                        # sum up activations of current batch in current layer of current encoder/decoder
                        batches_inside_layers_stacked[encoder_decoder_idx,layer_idx] += np.sum(np.array(activation_states[batch_key][encoder_decoder_key][layer_key].cpu()), axis=0)
                        
        if config.get("eval", "compute_in_parts") == "True":
            stacked = batches_inside_layers_stacked
        else:
            stacked = batches_inside_layers_stacked/num_samples
        stacked = stacked[:,:,:100,:] # only take activations for prompt

    else: # if model is Roberta or Bert
        # create a numpy array of shape (num_layers,num_batches, seq_len, hidden_size)
        num_samples = 0
        batches_inside_layers_stacked = np.zeros((model_config.num_hidden_layers,  
                                                  int(np.array(activation_states["batch 0"]["layer 0"][0].cpu()).shape[0]), 
                                                  model_config.intermediate_size))
        
        for batch_idx, batch_key in tqdm.tqdm(enumerate(activation_states.keys()), total=len(activation_states.keys())): # iterate over all batches
            num_samples += np.array(activation_states[batch_key]["layer 0"].cpu()).shape[0]
            for layer_idx, layer_key in enumerate(activation_states[batch_key].keys()): # iterate over all layers
            #for layer_idx, layer_key in enumerate(activation_states[i].keys()): # iterate over all layers
                # take mean across all activations for every sample of current batch in current layer
                batches_inside_layers_stacked[layer_idx] += np.sum(np.array(activation_states[batch_key][layer_key].cpu()), axis=0)

        if config.get("eval", "compute_in_parts") == "True":
            stacked = batches_inside_layers_stacked
        else:
            stacked = batches_inside_layers_stacked/num_samples
        
        stacked = stacked[:,:100,:] # only take activations for prompt

    return stacked


########################################
# tested #
########################################
def combine_a_bsl_parts(config):
    """Combine baseline activations sums of all parts and save to file.

    Args:
        config (dict): dictionary with config parameters of parsed config file

    Returns:
        None
    """

    number_parts = config.getint("eval", "total_parts")
    bsl_name = config.get("output", "model_name")
    seed = config.get("eval", "prompt_seed")
    a_bsl = np.array([np.load(f"./baseline_activations/{bsl_name}_{seed}/{bsl_name}_{seed}_{part}.npy").tolist() for part in range(1, number_parts)])
    a_bsl = np.sum(a_bsl, axis=0)
    np.save(f"./baseline_activations/{bsl_name}_{seed}.npy", a_bsl)

########################################
# tested #
########################################
def sum_to_mean(config, sum=False):
    """Compute mean of summed baseline activations and save to file.

    Args:
        config (dict): dictionary with config parameters of parsed config file

    Returns:
        None
    """
    number_parts = config.getint("eval", "total_parts")
    model_name = config.get("output", "model_name")
    seed = config.get("eval", "prompt_seed")
    if sum:
        a_bsl_sum = np.array([np.load(f"./baseline_activations_sum/{model_name}_{seed}/{model_name}_{seed}_{part}.npy").tolist() for part in range(1, number_parts)])
        a_bsl_sum = np.sum(a_bsl_sum, axis=0)
    else:
        a_bsl_sum = np.load(f"./baseline_activations_sum/{model_name}_{seed}.npy")
    num_samples = json.load(open(f"./samples_used/{model_name}.json", "r"))["num_samples"]
    a_bsl_mean = a_bsl_sum/num_samples
    np.save(f"./baseline_activations/{model_name}_{seed}.npy", a_bsl_mean)

#############################################################################################################
# tested #
#############################################################################################################
def compute_accuracy(activation_states_bsl, activation_states_val, data, config, model_config):
    """Sum up correct predictions for all neurons of activation_states_val depending on baseline activation and data['label'].
    
    Args:
        activation_states_bsl (numpy array): baseline activations of shape (num_layers, prompt_len, hidden_size) if bert/roberta or (2, num_layers, prompt_len, hidden_size) if t5
        activation_states_val (dict): dictionary with activations on one batch of validation data
        data (dict): dictionary with data of one batch
        config (dict): dictionary with config parameters of parsed config file
        model_config (dict): dictionary with config parameters of parsed model config file

    Returns:
        numpy array: array with shape (num_layers, prompt_len, hidden_size) if bert/roberta or (2, num_layers, prompt_len, hidden_size) if t5
    """

    if config.get("model","model_base") == "T5" :
        
        running_activations = np.zeros((2, model_config.num_layers, 100, model_config.d_ff)) # create a numpy array of shape (2,num_layers, prompt_len, hidden_size)
        data = data['label'].cpu().numpy().flatten() # get labels of batch, put to cpu and flatten

        for encoder_decoder_idx, encoder_decoder_key in enumerate(activation_states_val.keys()): # iterate over encoder and decoder
            for layer_idx, layer_key in enumerate(activation_states_val[encoder_decoder_key].keys()): # iterate over all layers
                #if np.array(activation_states_val[encoder_decoder_key][layer_key].cpu()).shape[1] == 230: # if encoder
                if encoder_decoder_key == "encoder":
                    activations = np.array(activation_states_val[encoder_decoder_key][layer_key].cpu()) # get activations of current layer and put to cpu
                    for i in range(activations.shape[0]): # iterate over all samples in batch
                        activation = activations[i, :100] # select only activations for prompt of current sample
                        inner_condition = activation > activation_states_bsl[encoder_decoder_idx][layer_idx] # creates array of shape (prompt_length, hidden_size) with true/false values, True where condition is met
                        outer_condition = inner_condition == data[i] # creates array of shape (prompt_length, hidden_size) with true/false values if result of inner condition is equal to label
                        accuracy = np.zeros((100,int(np.array(activation_states_val["encoder"]["layer 0"][0].cpu()).shape[1]))) # create array of shape (prompt_length, hidden_size)  
                        accuracy[outer_condition] = 1 # set all values to 1 where outer condition is met
                        running_activations[encoder_decoder_idx,layer_idx] += accuracy # add correct prediction of neuron to running_activations
    
    else: # if model is Roberta or Bert
        
        running_activations = np.zeros((model_config.num_hidden_layers, 100, model_config.intermediate_size)) # create numpy array of shape(num_layers, prompt_length, hidden_size)
        data = data['label'].cpu().numpy().flatten() # get labels of batch, put to cpu and flatten

        for layer_idx, layer_key in enumerate(activation_states_val.keys()):
            activations = np.array(activation_states_val[layer_key].cpu())
            for i in range(activations.shape[0]): # iterate over all samples in batch
                activation = activations[i, :100] # select only activations for prompt of current sample
                inner_condition = activation > activation_states_bsl[layer_idx] # creates array of shape (prompt_length, hidden_size) with true/false values, True where condition is met
                outer_condition = inner_condition == data[i]#[0] # creates array of shape (prompt_length, hidden_size) with true/false values if result of inner condition is equal to label
                accuracy = np.zeros((100,int(np.array(activation_states_val["layer 0"][0].cpu()).shape[1]))) # create array of shape (prompt_length, hidden_size)
                accuracy[outer_condition] = 1 # set all values to 1 where outer condition condition is met
                running_activations[layer_idx] += accuracy # add correct prediction of neuron to running_activations

    return running_activations     

#######################################################################################################################################
# review #
#######################################################################################################################################
def get_max_mean(model, dataset, seeds, mode="predictivity"):
    """Compute mean of max predictivities across all seeds for every neuron and save to file. This is the 6th formula of the skill neuron paper
    
    Args:
        model (str): model name
        dataset (str): dataset name
        seeds (list): list of seeds
        mode (str, optional): "predictivity" or "accuracy". Defaults to "predictivity" regarding which data to calculate with
    
    Returns:
        None
    """

    if mode == "predictivity":
        load_path = "./neuron_predictivities/" # shape(12,100,3072) if roberta, shape(2,12,100,3072) if t5
        save_path = "./skill_neuron_predictivities"
    else:
        load_path = "./neuron_accuracy/" # shape(12,100,3072) if roberta, shape(2,12,100,3072) if t5
        save_path = "./skill_neuron_accuracies"
    

    predictivities = np.array([np.load(f"{load_path}{dataset}Prompt{model}_{seed}.npy") for seed in seeds])
    
    if model == "T5":
        predictivities = predictivities[:,0] # only take encoder predictivities
    
    predictivities = np.max(predictivities, axis=2) # take max predictivity for every neuron across all prompt tokens
    predictivities = np.mean(predictivities, axis=0) # take mean across all seeds

    try:
        os.mkdir(save_path)
    except:
        pass

    np.save(f"{save_path}/{dataset}Prompt{model}_mean.npy", predictivities)

#######################################################################################################################################
# review #
#######################################################################################################################################
def get_max_mean_tuple(model, dataset, seeds, mode="predictivity"):
    """Compute mean of max predictivities across all seeds for every neuron and save to file. This is the 6th formula of the skill neuron paper
    
    Args:
        model (str): model name
        dataset (str): dataset name
        seeds (list): list of seeds
        mode (str, optional): "predictivity" or "accuracy". Defaults to "predictivity" regarding which data to calculate with
    
    Returns:
        None
    """

    if mode == "predictivity":
        load_path = "../../../../../../Volumes/Extreme SSD/Projects/20230501_bachelor_thesis/neuron_predictivities/" # shape(12,100,3072) if roberta, shape(2,12,100,3072) if t5
        save_path = "./neuron_predictivities_max_mean_tuple"
    else:
        load_path = "../../../../../../Volumes/Extreme SSD/Projects/20230501_bachelor_thesis/neuron_accuracy/" # shape(12,100,3072) if roberta, shape(2,12,100,3072) if t5
        save_path = "./neuron_accuracies_max_mean_tuple"
    

    predictivities = np.array([np.load(f"{load_path}{dataset}Prompt{model}_{seed}.npy") for seed in seeds])
    
    if model == "T5":
        predictivities = predictivities[:,0] # only take encoder predictivities
    
    arg_predictivities = np.argmax(predictivities, axis=2) # get index of token with max predictivity
    predictivities = np.max(predictivities, axis=2) # take max predictivity for every neuron across all prompt tokens
    predictivities = np.mean(predictivities, axis=0) # take mean across all seeds


    try:
        os.mkdir(save_path)
    except:
        pass

    np.save(f"{save_path}/{dataset}Prompt{model}_mean.npy", predictivities)

#######################################################################################################################################
# review #
#######################################################################################################################################
def get_mean_seed(model, dataset, seeds, mode="predictivity"):
    """Compute mean of max predictivities across all seeds for every neuron and save to file. This is the 6th formula of the skill neuron paper
    
    Args:
        model (str): model name
        dataset (str): dataset name
        seeds (list): list of seeds
        mode (str, optional): "predictivity" or "accuracy". Defaults to "predictivity" regarding which data to calculate with
    
    Returns:
        None
    """

    if mode == "predictivity":
        load_path = "../../../../../../Volumes/Extreme SSD/Projects/20230501_bachelor_thesis/neuron_predictivities_raw/" # shape(12,100,3072) if roberta, shape(2,12,100,3072) if t5
        save_path = "./neuron_predictivities_mean_dim_3"
    else:
        load_path = "../../../../../../Volumes/Extreme SSD/Projects/20230501_bachelor_thesis/neuron_accuracy_raw/" # shape(12,100,3072) if roberta, shape(2,12,100,3072) if t5
        save_path = "./neuron_accuracies_mean_dim_3"
    

    predictivities = np.array([np.load(f"{load_path}{dataset}Prompt{model}_{seed}.npy") for seed in seeds])
    
    if model == "T5":
        predictivities = predictivities[:,0] # only take encoder predictivities
    
    predictivities = np.mean(predictivities, axis=0) # take mean across all seeds

    try:
        os.mkdir(save_path)
    except:
        pass

    np.save(f"{save_path}/{dataset}Prompt{model}_mean.npy", predictivities)

def get_max_seed(model, dataset, seed, mode="predictivity"):
    """Compute max predictivities for one seed for every neuron and save to file. This is one half of the 6th formula of the skill neuron paper

    Args:
        model (str): model name
        dataset (str): dataset name
        seed (str): seed name
        mode (str, optional): "predictivity" or "accuracy". Defaults to "predictivity" regarding which data to calculate with

    Returns:
        None
    """


    if mode == "predictivity":
        load_path = "../../../../../../Volumes/Extreme SSD/Projects/20230501_bachelor_thesis/neuron_predictivities/"
        save_path = "./skill_neuron_predictivities/"
    else:
        load_path = "../../../../../../Volumes/Extreme SSD/Projects/20230501_bachelor_thesis/neuron_accuracy/"
        save_path = "./neuron_accuracies/"
    
    predictivities = np.load(f"{load_path}{dataset}Prompt{model}_{seed}.npy")
    
    if model == "T5":
        predictivities = predictivities[0] # only take encoder predictivities
    
    predictivities = np.max(predictivities, axis=1) # take max predictivity for every neuron across all prompt tokens

    try:
        os.mkdir(save_path)
    except:
        pass
    np.save(f"{save_path}{dataset}Prompt{model}_{seed}.npy", predictivities)



#############################################################################################################
# review #
#############################################################################################################
def max_skill_neurons_across_model(dataset, model, seed):
    """Sort neurons by predictivity after 6th formula in descending order and return 2-d array of indices where one index is [layer, neuron] 
    
    Args:
        dataset (str): dataset name
        model (str): model name
        seed (str): seed name

    Returns:
        numpy array: array with shape (num_neurons, 2)     
    """
    neuron_predictivity = np.load(f"/Users/leonackermann/Desktop/continuous_prompt_analysis/skill_neurons/neuron_predictivities/{dataset}Prompt{model}_{seed}.npy")
    neuron_predictivity_flat = neuron_predictivity.flatten() # flatten array
    sorted_indices_desc = np.argsort(neuron_predictivity_flat)[::-1] # sort neurons by predictivity in descending order
    reshaped_indices_desc = np.unravel_index(sorted_indices_desc, neuron_predictivity.shape) # reshape indices --> returns 2-tuple of arrays
    layer_indices = reshaped_indices_desc[0] # get layer indices
    neuron_indices = reshaped_indices_desc[1] # get neuron indices
    skill_neurons = [[layer_idx, neuron_idx] for layer_idx, neuron_idx in zip(layer_indices, neuron_indices)]

    skill_neurons = np.array(skill_neurons)

    try:
        os.mkdir(f"./skill_neurons_across_model/")
    except:
        pass
    np.save(f"./skill_neurons_across_model/{dataset}Prompt{model}_{seed}.npy", skill_neurons)

def create_random_neuron_predicitivities(dataset, model):
    """Create random neuron predictivities for whole model"""

    # Create a random NumPy array with values between 0.5 and 1
    lower_bound = 0.5
    upper_bound = 1.0
    shape = (12, 3072)
    random_neuron_predictivities = np.random.uniform(low=lower_bound, high=upper_bound, size=shape)
    try:
        os.mkdir(f"./random_neuron_predictivities")
    except:
        pass
    np.save(f"./random_neuron_predictivities/{dataset}Prompt{model}.npy", random_neuron_predictivities)

def create_random_neurons_across_model(dataset, model, dim=2):
    """Create random neurons for whole model"""
    #n = np.load(f"./skill_neurons_across_model_dim_{dim}/ethicsdeontologyPromptT5_33.npy").shape[0]
    n = np.load(f"./skill_neurons_across_model_dim_2/ethicsdeontologyPromptT5_33.npy").shape[0]
    rand_layers = np.random.randint(0, 12, size=n)
    rand_indices = np.random.randint(0, 3072, size=n)
    if dim == 3:
        if model == "T5":
            rand_tokens = np.random.randint(0, 230, size=n)
        else:
            rand_tokens = np.random.randint(0, 231, size=n) 
        random_neurons = np.stack([rand_layers, rand_tokens, rand_indices], axis=1)
    else:
        random_neurons = np.stack([rand_layers, rand_indices], axis=1)

    try:
        os.mkdir(f"./random_neurons_across_model_dim_{dim}/")
    except:
        pass

    np.save(f"./random_neurons_across_model_dim_{dim}/{dataset}Prompt{model}.npy", random_neurons)



#############################################################################################################
# review #
#############################################################################################################
def raw_skill_neurons_across_model(dataset, model, seed):
    """Sort neurons by their raw predictivity after 5th formula in descending order and return 2-d array of indices where one index is [layer, token, neuron] 
    
    Args:
        dataset (str): dataset name
        model (str): model name
        seed (str): seed name

    Returns:
        numpy array: array with shape (num_neurons, 3)     
    """
    load_path = "../../../../../../Volumes/Extreme SSD/Projects/20230501_bachelor_thesis/neuron_predictivities_raw/"
    neuron_predictivity = np.load(f"{load_path}{dataset}Prompt{model}_{seed}.npy")
    if model == "T5":
        neuron_predictivity = neuron_predictivity[0]
    neuron_predictivity_flat = neuron_predictivity.flatten() # flatten array
    sorted_indices_desc = np.argsort(neuron_predictivity_flat)[::-1] # sort neurons by predictivity in descending order
    reshaped_indices_desc = np.unravel_index(sorted_indices_desc, neuron_predictivity.shape) # reshape indices --> returns 3-tuple of arrays
    layer_indices = reshaped_indices_desc[0] # get layer indices
    prompt_indices = reshaped_indices_desc[1] # get prompt indices
    neuron_indices = reshaped_indices_desc[2] # get neuron indices
    skill_neurons = [[layer_idx, prompt_idx, neuron_idx] for layer_idx, prompt_idx,neuron_idx in zip(layer_indices, prompt_indices,neuron_indices)]

    skill_neurons = np.array(skill_neurons)

    try:
        os.mkdir(f"./raw_skill_neurons_across_model/")
    except:
        pass
    np.save(f"./raw_skill_neurons_across_model/{dataset}Prompt{model}_{seed}.npy", skill_neurons)

##############################################
# review #
##############################################
def skill_neurons_across_layer(dataset, model, seed="mean", type="skill_neuron"):
    """Sort neurons by predictivity in descending order within each layer

    Args:
        dataset (str): dataset name
        model (str): model name
        seed (str, optional): seed name. Defaults to "mean".

    Returns:
        numpy array: array with shape (12,3072)
    """
    if type == "skill_neuron":
        seed = f"_{seed}"
    else:
        seed = ""
    neuron_predictivities = np.load(f"./{type}_predictivities/{dataset}Prompt{model}{seed}.npy")
    neuron_indices = np.array([np.argsort(neuron_predictivities[i])[::-1] for i in range(12)])
    layer_indices = np.array([np.zeros(3072) + i for i in range(12)])
    skill_neurons = np.array([[[int(layer_idx), int(neuron_idx)] for layer_idx, neuron_idx in zip(layer_indices[i], neuron_indices[i])] for i in range(12)])

    try:
        os.mkdir(f"./{type}_across_layer/")
    except:
        pass
   
    np.save(f"./{type}_across_layer/{dataset}Prompt{model}{seed}.npy", skill_neurons)




# main 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+')
    parser.add_argument('--models', type=str, nargs='+')
    parser.add_argument('--prompt_seed', type=str, nargs='+')
    parser.add_argument('--compute_random_neurons', '-c', help="specific config file", default="False")
    parser.add_argument('--compute_skill_neurons', default="False")
    args = parser.parse_args()


    if args.compute_random_neurons == "True":
        for dataset in args.datasets:
            for model in args.models:
                skill_neurons_across_layer(dataset, model, "mean", "random_neuron")

    if args.compute_skill_neurons == "True":
        for dataset in args.datasets:
            for model in args.models:
                #get_max_mean(model, dataset, args.prompt_seed, mode="predictivity")
                skill_neurons_across_layer(dataset, model, "mean", "skill_neuron")