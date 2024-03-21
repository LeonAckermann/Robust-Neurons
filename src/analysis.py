import numpy as np
#import cupy as cp
import tqdm
import csv
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import copy
import json
import copy
import os
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import matplotlib.patches as mpatches
import argparse


def get_results(datasets, models, seeds):
    """Get accuracy results of model_seed for datasets and return dictionary and numpy array

    args:
        datasets: list of datasets
        models: list of models
        seeds: list of seeds

    returns:
        results: dictionary of results with keys: datasets, models, seeds
        results_arr: numpy array of results with shape (len(datasets), len(models), len(seeds))
    """

    results = {}
    results_arr = np.zeros((len(datasets), len(models), len(seeds)))
    for dataset in datasets:
        results[dataset] = {}
        for model in models:
            results[dataset][model] = {}
            for seed in seeds:
                seed = str(seed)
                path = f"./result/{dataset}Prompt{model}_{seed}/result.json"
                with open(path, "r") as f:
                    data = json.load(f)
                    data = json.loads(data)
                    acc = data["acc"]
                results_arr[datasets.index(dataset)][models.index(model)][seeds.index(seed)] = acc
                results[dataset][model][seed] = acc
                

    return results, results_arr

def get_adv_overlap(dataset, model, topn):
    predictivities = np.load(f"/Users/leonackermann/Desktop/continuous_prompt_analysis/skill_neurons/skill_neuron_predictivities/{dataset}Prompt{model}_mean.npy")
    predictivities_adv = np.load(f"/Users/leonackermann/Desktop/continuous_prompt_analysis/skill_neurons/skill_neuron_predictivities/Adv{dataset}Prompt{model}_mean.npy")
    predictivities = np.argsort(predictivities, axis=1)[::-1]
    predictivities_adv = np.argsort(predictivities_adv, axis=1)[::-1]
    percent = int(predictivities.shape[1]/100*topn)
    predictivities = predictivities[:,:percent]
    predictivities_adv = predictivities_adv[:,:percent]
    overlap = np.zeros((12))
    for i in range(12):
        set1 = set(predictivities[i].tolist())
        set2 = set(predictivities_adv[i])
        overlap[i] = len(set1.intersection(set2))/predictivities_adv.shape[1]

    return overlap

def get_adv_overlap_per_topn(datasets, models, topn):
    topn_overlap = {}
    for dataset in datasets:
        topn_overlap[dataset] = {}
        for model in models:
            topn_overlap[dataset][model] = []
            for i in range(2,topn +1,2):
                
                overlap = get_adv_overlap(dataset, model, i)
                topn_overlap[dataset][model].append([np.mean(overlap), get_sem(overlap)])
                    

    return topn_overlap

def get_correlation_per_topn(datasets, models, topn):
    topn_correlation = {}
    for dataset in datasets:
        topn_correlation[dataset] = {}
        for model in models:
            topn_correlation[dataset][model] = []
            for i in range(2,topn +1,2):
                
                correlation = get_correlation_set(dataset, model, i)
                topn_correlation[dataset][model].append([np.mean(correlation), get_sem(correlation)])
                    

    return topn_correlation

def get_dataset_correlation_per_topn(tasks, models, topn):
    topn_correlation = {}
    for task in tasks:
        topn_correlation[task] = {}
        for model in models:
            topn_correlation[task][model] = []
            for i in range(2,topn +1,2):
                
                correlation = get_task_correlation_set(task, model, i)
                if task=="average" or task == "sentiment_analysis":
                    topn_correlation[task][model].append([np.mean(correlation), get_sem(correlation)])
                else:
                    topn_correlation[task][model].append([correlation, 0])
                    

    return topn_correlation


def get_correlation_set(dataset, model, topn):
    matrix = np.load(f"/Users/leonackermann/Desktop/continuous_prompt_analysis/skill_neurons/plots/correlation_plots_final/seeds/topn/topn-{topn}/{dataset}Prompt{model}_topn-{topn}.npy")
    n = matrix.shape[0]
    i = 0
    values = np.zeros((10))
    for j in range(4):
        for k in range(j+1,5):
            values[i] = matrix[j,k]
            i += 1

def get_task_correlation_set(task, model, topn):
    matrix = np.load(f"/Users/leonackermann/Desktop/continuous_prompt_analysis/skill_neurons/plots/correlation_plots_final/predictivity/topn-{topn}/{model}_mean_across_layer_all.npy")
    n = matrix.shape[0]
    i = 0
    if task=="ethical_judgment":
        values = np.zeros((1))
        values = matrix[0,1]
    elif task == "paraphrase_identification":
        values = np.zeros((1))
        values = matrix[2,3]
    elif task == "sentiment_analysis":
        values = np.zeros((3))
        values[0] = matrix[7,8]
        values[1] = matrix[7,9]
        values[2] = matrix[8,9]
    elif task == "average":
        values = np.zeros((55))
        for j in range(10):
            for k in range(j+1,11):
                values[i] = matrix[j,k]
                i += 1
    return values

def get_matrix_correlation_set(matrix):
    n = matrix.shape[0]
    i = 0
    
    values = np.zeros((55))
    for j in range(10):
        for k in range(j+1,11):
            values[i] = matrix[j,k]
            i += 1
    return values
   

def get_mean_std(datasets,models,seeds):
    results_dict, _ = get_results(datasets, models, seeds)
    new_dict = {}
    # iterate over keys
    for dataset in results_dict.keys():
        new_dict[dataset] = {}
        for models in results_dict[dataset].keys():
            new_dict[dataset][models] = {}
            new_dict[dataset][models]["mean"] = np.mean(list(results_dict[dataset][models].values()))*100
            new_dict[dataset][models]["std"] = np.std(list(results_dict[dataset][models].values()))*100

    flat_data = {}
    for task, models in new_dict.items():
        for model, values in models.items():
            model_key = f'{model}_mean'
            std_key = f'{model}_std'
            if model_key not in flat_data:
                flat_data[model_key] = []
                flat_data[std_key] = []
            flat_data[model_key].append(values['mean'])
            flat_data[std_key].append(values['std'])

    # Create a pandas DataFrame
    df = pd.DataFrame(flat_data, index=list(new_dict.keys()))

    return df

def get_sem(data):
    """ Calculate the standard error of the mean (SEM) for the given data.
    
    args:
        data: list of accuracies of one model on one dataset with different seeds, shape (n,)

    returns:
        sem: standard error of the mean
    """
    s = np.std(data, ddof=1)  # ddof=1 for unbiased estimation
    n = len(data)
    sem = s / np.sqrt(n)  # Calculate the standard error of the mean
    return sem

def get_perturbation_accuracy_runs(datasets, models, perturbation, type,seeds, mode, layer=False, adv_nonAdv = "_sn"):
    """Get accuracy results of model_seed for datasets with up to {perturbation} percent perturbed neurons and return dictionary and numpy array

    args:
        datasets: list of datasets
        models: list of models
        perturbation: int, maximum of perturbation in percent
        type: "random_neurons" or "skill_neurons"
        seeds: list of seeds

    returns:
        perturbation_accuracy: dictionary of results with keys: datasets, models, perturbation
    """
    if layer==True:
        layer_model_mode = "_layer"
    else:
        layer_model_mode = ""
    perturbation_accuracy = {}
    for dataset in datasets:
        perturbation_accuracy[dataset] = {}
        for model in models:
            perturbation_accuracy[dataset][model] = []
            #percent =[1,25,50,75,100]
            for i in [0,1,3,5,7,9,11,13,15]:
            #for i in percent:
                n = i/10
                if n == 0.2:
                    n = "0_2"
                elif n== 0.4:
                    n = "0_4"
                elif n== 0.6:
                    n = "0_6"
                elif n== 0.8:
                    n = "0_8"
                elif n== 1.0:
                    n = "1"
                accuracy = []
                for seed in seeds:
                    if False == True:
                        break
                    #if i == 0:
                    #    load_path = f"../../../../../../Volumes/Extreme SSD/Projects/20230501_bachelor_thesis/result2_suppression_{type}_max_mean_230/{dataset}Prompt{model}_{seed}{adv_nonAdv}_topn-{i}/result.json"
                    else:
                        #load_path = f"../../../../../../Volumes/Extreme SSD/Projects/20230501_bachelor_thesis/result2_suppression_{type}_max_mean_230/{dataset}Prompt{model}_{seed}_topn-{i}/result.json"
                        if "Adv" in dataset:
                            load_path = f"../../../../../../Volumes/Extreme SSD/Projects/20230501_bachelor_thesis/result2_suppression_{type}_max_mean_230/{dataset}Prompt{model}_{seed}_sn_{dataset[3:]}_topn-{i}/result.json"
                            #load_path = f"../../../../../../Volumes/Extreme SSD/Projects/20230501_bachelor_thesis/result2_suppression_{type}_max_mean_230/{dataset}Prompt{model}_{seed}{adv_nonAdv}_topn-{i}/result.json"
                        else:
                            load_path = f"../../../../../../Volumes/Extreme SSD/Projects/20230501_bachelor_thesis/result2_suppression_{type}_max_mean_230/{dataset}Prompt{model}_{seed}_sn_{dataset}_topn-{i}/result.json"
                            #load_path = f"../../../../../../Volumes/Extreme SSD/Projects/20230501_bachelor_thesis/result2_suppression_{type}_max_mean_230/{dataset}Prompt{model}_{seed}{adv_nonAdv}_topn-{i}/result.json"
                    print("load_path", load_path)
                    with open(load_path, "r") as f:
                        data = json.load(f)
                        data = json.loads(data)
                        accuracy.append(data["acc"])
                accuracy = np.array(accuracy)
                perturbation_accuracy[dataset][model].append([np.mean(accuracy), get_sem(accuracy)])
                    

    return perturbation_accuracy


def get_transferability_matrix_old(datasets, model, seeds, relative=True):
    n = len(datasets)
    matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if "Adv" not in datasets[j]:
                accuracy = []
                for seed in seeds:
                    with open(f"../../../../../../Volumes/Extreme SSD/Projects/20230501_bachelor_thesis/result_transferability/{datasets[i]}Prompt{model}_{datasets[j]}_{seed}/result.json", "r") as f:
                        data = json.load(f)
                        data = json.loads(data)
                        accuracy.append(data["acc"])
                accuracy = np.mean(np.array(accuracy))
                if relative==True and "Adv" not in datasets[i]:
                    relative_accuracy = []
                    for seed in seeds:
                        with open(f"../../../../../../Volumes/Extreme SSD/Projects/20230501_bachelor_thesis/result_transferability/{datasets[i]}Prompt{model}_{datasets[i]}_{seed}/result.json", "r") as f:
                            data = json.load(f)
                            data = json.loads(data)
                            relative_accuracy.append(data["acc"])
                    relative_accuracy = np.mean(np.array(relative_accuracy))
                    accuracy = int(accuracy/relative_accuracy*100)
                elif relative==True and "Adv" in datasets[i]:
                    accuracy = None
                elif relative==False:
                    accuracy = int(accuracy*100)
            else:
                accuracy = None
            matrix[i,j] = accuracy

    return matrix


def get_transferability_matrix_new(datasets, model, seeds, relative=True):
    n = len(datasets)
    matrix = np.zeros((n,n))
    for i in range(n): # iterate over target task
        for j in range(n): # iterate over source task
            if "Adv" not in datasets[j]:
                accuracy = []
                for seed in seeds:
                    with open(f"../../../../../../Volumes/Extreme SSD/Projects/20230501_bachelor_thesis/result_transferability/{datasets[i]}Prompt{model}_{datasets[j]}_{seed}/result.json", "r") as f:
                        data = json.load(f)
                        data = json.loads(data)
                        accuracy.append(data["acc"])
                if relative==True and "Adv" not in datasets[i]:
                    accuracy = np.array(accuracy)
                else:
                    accuracy = np.mean(np.array(accuracy))
                if relative==True and "Adv" not in datasets[i]:
                    relative_accuracy = []
                    for seed in seeds:
                        with open(f"../../../../../../Volumes/Extreme SSD/Projects/20230501_bachelor_thesis/result_transferability/{datasets[i]}Prompt{model}_{datasets[i]}_{seed}/result.json", "r") as f:
                            data = json.load(f)
                            data = json.loads(data)
                            relative_accuracy.append(data["acc"])
                    relative_accuracy = accuracy/np.array(relative_accuracy)*100
                    accuracy = np.mean(relative_accuracy)
                    #accuracy = int(accuracy/relative_accuracy*100)
                elif relative==True and "Adv" in datasets[i]:
                    accuracy = None
                elif relative==False:
                    accuracy = int(accuracy*100)
            else:
                accuracy = None
            matrix[j,i] = accuracy

    return matrix


def correlation_matrix_across_model(datasets, model):
    """Compute correlation matrix of mean skill neurons sorted across all layers for multiple datasets and one model

    args:
        datasets: list of datasets
        model: model

    reutrns:
        matrix: correlation matrix of shape (len(datasets), len(datasets)) 
    """
    n = len(datasets)
    skill_neurons = np.array([np.load(f"./skill_neurons_across_model_dim_2/{dataset}Prompt{model}.npy") for dataset in datasets])
    skill_neurons += 1
    skill_neurons = np.prod(skill_neurons, axis=2)
    
    matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = spearmanr(skill_neurons[i], skill_neurons[j])[0]
    
    return matrix


def correlation_matrix_across_seed(dataset, model, seeds, topn=None):
    """Compute correlation matrix for seed specific skill neurons of one dataset and one model
    
    args:
        dataset: dataset
        model: model
        seeds: list of seeds
        topn: 
            - if None, compute mean of correlation of seed specific neuron predictivites for each layer between seeds
            - else compute correlation matrix for topn percent skill neurons sorted across all layers for each seed 

    returns:
        matrix: correlation matrix of shape (len(seeds), len(seeds))
    
    """
    n = len(seeds)
    if topn==None:
        skill_neurons = [np.load(f"./skill_neuron_predictivities/{dataset}Prompt{model}_{seed}.npy") for seed in seeds]
        matrix = np.zeros((12,n,n))
        for k in range(12):
            for i in range(n):
                for j in range(n):
                    matrix[k][i][j] = spearmanr(skill_neurons[i][k], skill_neurons[j][k])[1]

        return np.mean(matrix, axis=0)
    
    else:
        #skill_neurons = np.array([np.load(f"/Users/leonackermann/Desktop/continuous_prompt_analysis/skill_neurons/skill_neurons_across_model_dim_2/{dataset}Prompt{model}_{seed}.npy") for seed in seeds])
        mean_predictivities = np.load(f"/Users/leonackermann/Desktop/continuous_prompt_analysis/skill_neurons/skill_neurons_across_layer/{dataset}Prompt{model}_mean.npy")
        skill_neurons = np.array([np.load(f"/Users/leonackermann/Desktop/continuous_prompt_analysis/skill_neurons/skill_neuron_across_layer/{dataset}Prompt{model}_{seed}.npy") for seed in seeds])
        neuron_predictivities = [np.load(f"/Users/leonackermann/Desktop/continuous_prompt_analysis/skill_neurons/skill_neuron_predictivities/{dataset}Prompt{model}_{seed}.npy") for seed in seeds]

        
        #neuron_predictivities = np.array([np.sort(neuron_predictivities[i])[:,::-1] for i in range(len(neuron_predictivities))])
        #number_neurons = len(skill_neurons[0].flatten())
        number_neurons = 3072
        percent = int(number_neurons/100*topn)
        skill_neurons = np.array([skill_neurons[i,:,:percent] for i in range(len(neuron_predictivities))])
        top_skill_neurons = np.zeros((len(seeds),12,percent))
        for i in range(len(seeds)):
            for j in range(12):
                l = 0
                temp = skill_neurons[i,j,:percent]
                temp = [temp[m,1] for m in range(len(temp))]
                for k in range(3072):
                    if k in temp:
                        top_skill_neurons[i,j,l] = neuron_predictivities[i][j][k]
                        l+=1

        matrix = np.zeros((12,n,n))
        for k in range(12):
            for i in range(n):
                for j in range(n):
                    matrix[k][i][j] = spearmanr(top_skill_neurons[i][k], top_skill_neurons[j][k])[0]

        return np.mean(matrix, axis=0)
        

def correlation_matrix_across_layer(datasets, model,data,layer="across_layer_all",seed="mean", topn=100, normalize=False):
    """Compute correlation matrix of seed specific or mean neuron predictivities, neuron accuracies or skill neurons for one layer or all layers
    
    args:
        datasets: list of datasets
        model: model
        data: "predictivity", "accuracy" or "skill_neurons"
        layer: "across_layer_all" or "across_layer_{layer_number}"
        seed: "mean" or seed number

    returns:
        matrix: 
            - correlation matrix of shape (len(datasets), len(datasets))
            - if layer=="across_layer_all", matrix is mean of correlation matrix of each layer, else matrix is correlation matrix of one layer
            - if seed=="mean", matrix is correlation matrix of mean neuron predictivities/accuracies/skill neurons of each seed, else matrix is correlation matrix of one seed 

    """
    n = len(datasets)
    #if seed!=None:
    #    seed = f"_{seed}"
    #else:
    #    seed = ""
    #if topn==100:
    if False == True:
        if data == "predictivity":
            skill_neurons = [np.load(f"./neuron_predictivities/{dataset}Prompt{model}_{seed}.npy") for dataset in datasets]
        elif data == "accuracy":
            skill_neurons = [np.load(f"./neuron_accuracies/{dataset}Prompt{model}_{seed}.npy") for dataset in datasets]
        elif data =="skill_neurons":
            skill_neurons = [np.load(f"./skill_neurons_unsorted/{dataset}Prompt{model}_{seed}.npy") for dataset in datasets]

    else:
        skill_neurons = np.array([np.load(f"/Users/leonackermann/Desktop/Projects/continuous_prompt_analysis/skill_neurons/skill_neuron_across_layer/{dataset}Prompt{model}_{seed}.npy") for dataset in datasets])
        neuron_predictivities = [np.load(f"/Users/leonackermann/Desktop/Projects/continuous_prompt_analysis/skill_neurons/skill_neuron_predictivities/{dataset}Prompt{model}_{seed}.npy") for dataset in datasets]

        
        #neuron_predictivities = np.array([np.sort(neuron_predictivities[i])[:,::-1] for i in range(len(neuron_predictivities))])
        #number_neurons = len(skill_neurons[0].flatten())
        number_neurons = 3072
        percent = int(number_neurons/100*topn)
        skill_neurons = np.array([skill_neurons[i,:,:percent] for i in range(len(neuron_predictivities))])
        top_skill_neurons = np.zeros((len(datasets),12,percent))
        for i in range(len(datasets)):
            for j in range(12):
                l = 0
                temp = skill_neurons[i,j,:percent]
                temp = [temp[m,1] for m in range(len(temp))] # only take neuron index from temp
                for k in range(3072):
                    if k in temp:
                        top_skill_neurons[i,j,l] = neuron_predictivities[i][j][k]
                        l+=1
        skill_neurons = top_skill_neurons

    matrix = np.zeros((12,n,n))
    p_matrix = np.zeros((12,n,n))
    for k in range(12):
        for i in range(n):
            for j in range(n):
                matrix[k][i][j] = spearmanr(skill_neurons[i][k][:], skill_neurons[j][k][:])[0]
                p_matrix[k][i][j] = spearmanr(skill_neurons[i][k][:], skill_neurons[j][k][:])[1]
    
    if layer == "across_layer_all":
        matrix = np.mean(matrix, axis=0)
        p_matrix = np.mean(p_matrix, axis=0)
        if normalize==True:
            normalized = get_matrix_correlation_set(matrix)
            for i in range(n):
                for j in range(n):
                    matrix[i][j] = (matrix[i][j]-np.mean(normalized))/np.std(normalized)
        return matrix, p_matrix
    return matrix[int(layer[13:])]


def plot_transferability_matrix(datasets, model, seeds, relative=True):
    """Plot correlation matrix of shape(len(datasets), len(datasets)) which is generated depending on args, x-axis: datasets, y-axis: datasets

    args:
        datasets: list of datasets
        model: model
        data: "predictivity", "accuracy" or "skill_neurons"
        mode: "across_model" or "across_layer_all" or "across_layer_{layer_number}"
        title: title of plot
        seed: seed number or mean

    returns:
        None    
    """
    
    matrix = get_transferability_matrix_new(datasets, model, seeds, relative)
    print(matrix)
    print(np.max(matrix))
    fig, ax = plt.subplots()
    #im = ax.imshow(matrix)

    #cmap = color #plt.cm.GnBu
    cmap = "viridis"
    #cmap = 'viridis'
    #cmap = 'cividis'

    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=np.nanmax(matrix))

    cbar = ax.figure.colorbar(im, ax=ax, location='right')
    cbar.ax.set_ylabel("Performance", rotation=-90, va="bottom", fontsize=14)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(datasets)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_yticklabels(datasets, fontsize=12)

    #tick_colors = ["mediumaquamarine","mediumaquamarine", "blueviolet", "blueviolet","blueviolet","blueviolet","blueviolet","steelblue","steelblue","steelblue","steelblue"]
    #tick_colors = ["darkgrey", "darkgrey", "gold", "gold", "gold", "gold", "gold", "crimson", "crimson", "crimson", "crimson"]
    if relative == False:
        tick_colors = ["midnightblue", "midnightblue", "darkmagenta", "darkmagenta", "darkmagenta", "blueviolet", "blueviolet","blueviolet", "blueviolet", "darkred", "darkred", "darkred", "darkred"]
    else:
        tick_colors = ["midnightblue", "midnightblue", "darkmagenta", "darkmagenta", "blueviolet","blueviolet","blueviolet","blueviolet", "darkred", "darkred", "darkred"]
    tick_colors = ["midnightblue", "midnightblue", "darkmagenta", "darkmagenta", "blueviolet", "darkred", "darkred", "darkred"]
    for i, ticklabel in enumerate(ax.get_xticklabels()):
        ticklabel.set_color(tick_colors[i])
    for i, ticklabel in enumerate(ax.get_yticklabels()):
        ticklabel.set_color(tick_colors[i])
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    
    #Loop over data dimensions and create text annotations.
    for i in range(len(datasets)):
        for j in range(len(datasets)):
            if np.isnan(matrix[i,j]) == False:
                text = ax.text(j, i, int(matrix[i, j]),
                            ha="center", va="center", color="black", fontsize=10)


    #ax.set_title("Correlation between skill neurons")
    #ax.set_title(title, pad=20)
    ax.set_xlabel("Target Task", fontsize=14)
    ax.set_ylabel("Source Task", fontsize=14)
    fig.tight_layout()

    #if seed != None:
    #    seed = f"_{seed}"
    #else:
    #    seed = ""
    
    #create image out of fiture
    if relative==True:
        fig.savefig(f"./plots/transferability/relative/{model}.png", dpi=600)
    else:
        fig.savefig(f"./plots/transferability/absolute/{model}.png", dpi=600)


    

    plt.close()


def plot_matrix_across_seeds(datasets, model, seeds, topn=None, mean=False, adversarial=False):
    """Plot correlation matrix for 

    args:
        datasets: list of datasets
        model: model
        seeds: list of seeds
        topn: None if all neurons are used, else topn percent of neurons are used
        mean: if True plot mean of correlation matrix of all datasets, else plot correlation matrix for each datasets
        adversarial: used together with mean==True, just changes the save_path of the plot

    returns:
        None
    """

    matrices = []
    for dataset in datasets:
        matrix = correlation_matrix_across_seed(dataset, model, seeds, topn)
        matrices.append(matrix)

        if mean==False:
            fig, ax = plt.subplots()
            im = ax.imshow(matrix)
            cmap = plt.cm.GnBu
            im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)
            cbar = ax.figure.colorbar(im, ax=ax, location='right')
            cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")

            # We want to show all ticks...
            ax.set_xticks(np.arange(len(seeds)))
            ax.set_yticks(np.arange(len(seeds)))
            # ... and label them with the respective list entries
            ax.set_xticklabels(seeds)
            ax.set_yticklabels(seeds)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                    rotation_mode="anchor")

            #Loop over data dimensions and create text annotations.
            for i in range(len(seeds)):
                for j in range(len(seeds)):
                    text = ax.text(j, i, round(matrix[i, j],5),
                                ha="center", va="center", color="w", fonsize=5)
            #ax.set_title("Correlation between skill neurons")

            #if topn!=None:    
            #    ax.set_title(f"Correlation of top {int(topn)*2} percent skill neurons of {model} on {dataset}", pad=20)
            #else:
            #    ax.set_title(f"Correlation of neuron predictivities of {model} on {dataset} across all seeds", pad=20)
            fig.tight_layout() 

            if adversarial==True:
                adversarial = "_adversarial"
            else:
                adversarial = ""

            if topn==None:
                try:
                    os.mkdir(f"./plots/correlation_plots_final/seeds/neuron_predictivities")
                except:
                    pass

                fig.savefig(f"./plots/correlation_plots_final/seeds/neuron_predictivities/{dataset}Prompt{model}{adversarial}.png")
                np.save(f"./plots/correlation_plots_final/seeds/neuron_predictivities/{dataset}Prompt{model}{adversarial}.npy", matrix)
            else:
                try:
                    os.mkdir(f"./plots/correlation_plots_final/seeds/topn")
                except:
                    pass

                try:
                    os.mkdir(f"./plots/correlation_plots_final/seeds/topn/topn-{topn}")
                except:
                    pass
                fig.savefig(f"./plots/correlation_plots_final/seeds/topn/topn-{topn}/{dataset}Prompt{model}_topn-{topn}{adversarial}.png")
                np.save(f"./plots/correlation_plots_final/seeds/topn/topn-{topn}/{dataset}Prompt{model}_topn-{topn}{adversarial}.npy", matrix)

            plt.close()
    
    if mean==True:
        matrices = np.array(matrices)
        #print(matrices.shape)
        matrices = np.mean(matrices, axis=0)
        print(matrices.shape)
        fig, ax = plt.subplots()
        im = ax.imshow(matrix)
        cmap = plt.cm.GnBu
        im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)
        cbar = ax.figure.colorbar(im, ax=ax, location='right')
        cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")
        # We want to show all ticks...
        ax.set_xticks(np.arange(len(seeds)))
        ax.set_yticks(np.arange(len(seeds)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(seeds)
        ax.set_yticklabels(seeds)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
        #Loop over data dimensions and create text annotations.
        for i in range(len(seeds)):
            for j in range(len(seeds)):
                text = ax.text(j, i, round(matrix[i, j],2),
                            ha="center", va="center", color="w", fontsize=10)
        #ax.set_title("Correlation between skill neurons")
        #if topn!=None:    
        #    ax.set_title(f"Correlation of top {int(topn)*2} percent skill neurons of {model} on {dataset}", pad=20)
        #else:
        #    ax.set_title(f"Correlation of neuron predictivities of {model} on {dataset} across all seeds", pad=20)
        fig.tight_layout() 

        if adversarial==True:
            adversarial = "_adversarial"
        else:
           adversarial = ""
        if topn==None:

            try:
                os.mkdir(f"./plots/correlation_plots_final/seeds/neuron_predictivities")
            except:
                pass
            fig.savefig(f"./plots/correlation_plots_final/seeds/neuron_predictivities/meanPrompt{model}{adversarial}.png")
            np.save(f"./plots/correlation_plots_final/seeds/neuron_predictivities/meanPrompt{model}{adversarial}.npy", matrices)
        else:
            try:
                os.mkdir(f"./plots/correlation_plots_final/seeds/skill_neurons_across_model")
            except:
                pass
            try:
                os.mkdir(f"./plots/correlation_plots_final/seeds/skill_neurons_across_model/topn-{topn}")
            except:
                pass
            fig.savefig(f"./plots/correlation_plots_final/seeds/topn/topn-{topn}/meanPrompt{model}_topn-{topn}_mean{adversarial}.png")
            np.save(f"./plots/correlation_plots_final/seeds/topn/topn-{topn}/meanPrompt{model}_topn-{topn}_mean{adversarial}.npy", matrices)
        plt.close()
        



def plot_matrix(datasets, model, data,mode, title,seed="mean",topn=100, normalized=False):
    """Plot correlation matrix of shape(len(datasets), len(datasets)) which is generated depending on args, x-axis: datasets, y-axis: datasets

    args:
        datasets: list of datasets
        model: model
        data: "predictivity", "accuracy" or "skill_neurons"
        mode: "across_model" or "across_layer_all" or "across_layer_{layer_number}"
        title: title of plot
        seed: seed number or mean

    returns:
        None    
    """
    
    if mode=="across_model":
        matrix = correlation_matrix_across_model(datasets, model)
    else:
        matrix, p_matrix = correlation_matrix_across_layer(datasets, model, data, mode, seed, topn, normalized)

    #matrix = p_matrix
    fig, ax = plt.subplots()
    im = ax.imshow(matrix)

    #cmap = color #plt.cm.GnBu
    #cmap = plt.cm.GnBu
    cmap = 'viridis'
    #cmap = 'cividis'
    if normalized==True:
        im = ax.imshow(matrix, cmap=cmap, vmin=np.min(matrix), vmax=np.max(matrix))
    else:
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1)


    cbar = ax.figure.colorbar(im, ax=ax, location='right')
    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom", fontsize=18)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(datasets)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_yticklabels(datasets, fontsize=12)

    #tick_colors = ["mediumaquamarine","mediumaquamarine", "blueviolet", "blueviolet","blueviolet","blueviolet","blueviolet","steelblue","steelblue","steelblue","steelblue"]
    #tick_colors = ["darkgrey", "darkgrey", "gold", "gold", "gold", "gold", "gold", "crimson", "crimson", "crimson", "crimson"]
    tick_colors = ["midnightblue", "midnightblue", "darkmagenta", "darkmagenta", "darkmagenta", "blueviolet", "blueviolet", "blueviolet", "blueviolet","darkred", "darkred", "darkred", "darkred"]
    #tick_colors = ["midnightblue", "midnightblue", "darkmagenta", "darkmagenta", "blueviolet", "blueviolet"]
    for i, ticklabel in enumerate(ax.get_xticklabels()):
        ticklabel.set_color(tick_colors[i])
    for i, ticklabel in enumerate(ax.get_yticklabels()):
        ticklabel.set_color(tick_colors[i])
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    
    #Loop over data dimensions and create text annotations.
    #if normalized ==True or normalized==False:
    #    for i in range(len(datasets)):
    #        for j in range(len(datasets)):
    #            text = ax.text(j, i, np.round(matrix[i, j],5),
    #                        ha="center", va="center", color="white", fontsize=4)


    #ax.set_title("Correlation between skill neurons")
    #ax.set_title(title, pad=20)
    fig.tight_layout()

    #if seed != None:
    #    seed = f"_{seed}"
    #else:
    #    seed = ""
    
    #fig.tight_layout() 
    #create image out of fiture
    if data=="predictivity":
        if normalized==True:
            save_path = "./plots/correlation_plots_final/predictivity/normalized"
        else:
            save_path = "./plots/correlation_plots_final/predictivity"
        try:
            os.mkdir(f"{save_path}/topn-{topn}")
        except:
            pass
        np.save(f"{save_path}/topn-{topn}/{model}_{seed}_{mode}.npy", matrix)
        fig.savefig(f"{save_path}/topn-{topn}/{model}_{seed}_{mode}.png", dpi=600)
    elif data=="accuracy":
        fig.savefig(f"./plots/correlation_plots_final/accuracy/{model}_{seed}_{mode}.png")
    elif data=="skill_neurons":
        fig.savefig(f"./plots/correlation_plots_final/mean/{model}_{seed}_{mode}.png")
    elif data=="argmax":
        fig.savefig(f"./plots/correlation_plots_final/argmax/{model}_{seed}_{mode}.png")
    else:
        fig.savefig(f"./plots/correlation_plots_final/seed_{seed}/{model}_mode_{mode}.png")

    plt.close()

def get_bin_boundaries(data):
    """Get bin boundaries for histogram plot of data

    args:
        data: list of neuron predictivities, shape (n,)

    returns:
        bin_boundaries: list of bin boundaries
    """
    maximum = np.max(data)
    minimum = np.min(data)
    minimum = np.floor(minimum / 0.05) * 0.05
    maximum = np.ceil(maximum / 0.05) * 0.05
    bin_boundaries = np.arange(minimum, maximum+0.05, 0.05)
    return bin_boundaries

def get_predictivities_adversarial_skill_neurons_nonadversarial(dataset, model, topn):
    predictivities_adversarial = np.load(f"/Users/leonackermann/Desktop/continuous_prompt_analysis/skill_neurons/skill_neuron_predictivities/Adv{dataset}Prompt{model}_mean.npy")  
    skill_neurons_nonadversarial = np.load(f"/Users/leonackermann/Desktop/continuous_prompt_analysis/skill_neurons/skill_neurons_across_model_dim_2/{dataset}Prompt{model}_mean.npy")
    percent = int(skill_neurons_nonadversarial.shape[0]/100*topn)
    skill_neurons_nonadversarial = skill_neurons_nonadversarial[:percent]
    predictivities_adversarial_topn = np.zeros(skill_neurons_nonadversarial.shape)
    #for i in range(len(skill_neurons_nonadversarial)):
    for i, neuron in enumerate(skill_neurons_nonadversarial):
        layer_idx, neuron_idx = neuron
        predictivities_adversarial_topn[i] = predictivities_adversarial[layer_idx,neuron_idx]
    return predictivities_adversarial_topn

def get_histplot_distribution(datasets, models, topn, per_layer, mode, seed, skill=True, location=False, adversarial=False):

    if mode=="mean_seeds":
        load_path="/Users/leonackermann/Desktop/continuous_prompt_analysis/skill_neurons/neuron_predictivities_mean_dim_3"
    else:
        load_path="/Users/leonackermann/Desktop/continuous_prompt_analysis/skill_neurons/neuron_predictivities"
    for dataset in datasets:
        sorted_list = []
        for model in models:
            if adversarial ==False:
                predictivities = np.load(f"{load_path}/{dataset}Prompt{model}_{seed}.npy")
            else:
                predictivities = get_predictivities_adversarial_skill_neurons_nonadversarial(dataset, model, topn)
            if skill==True:
                skill_neurons = np.load(f"/Users/leonackermann/Desktop/continuous_prompt_analysis/skill_neurons/skill_neurons_across_model_dim_2/{dataset}Prompt{model}_mean.npy")
                save_type = "skill_neurons"
            else:
                skill_neurons = np.load(f"/Users/leonackermann/Desktop/continuous_prompt_analysis/skill_neurons/random_neurons_across_model/{dataset}Prompt{model}.npy")
                save_type = "random_neurons"
                print("random_neurons")
            seed_dict = {"unknown": 0, "33":1, "34":2, "35":3, "36":4}
            seeds = ["unknown", 33, 34, 35, 36]
            #predictivities_raw = np.array([np.load(f"../../../../../../Volumes/Extreme SSD/Projects/20230501_bachelor_thesis/neuron_predictivities_raw/{dataset}Prompt{model}_{seed}.npy") for seed in seeds]) # shape(12,100,3072) if roberta, shape(2,12,100,3072) if t5
            #predictivities_mean_dim_3 = np.load(f"/Users/leonackermann/Desktop/continuous_prompt_analysis/skill_neurons/neuron_predictivities_mean_dim_3/{dataset}Prompt{model}_mean.npy")
            if per_layer == True:
                for i in range(len(predictivities)):
                    sorted = np.sort(predictivities[i].flatten())
                    bin_boundaries = get_bin_boundaries(sorted)
                    print()
                    num_bins = np.arange(0, len(sorted), 100000)
                    fig, ax = plt.subplots(figsize=(10,7))
                    ax.hist(sorted,bins=bin_boundaries)
                    ax.set_title(f"Distribution of predictivities for {dataset}Prompt{model} layer {i}")
                    ax.set_xlabel("Predictivity")
                    ax.set_ylabel("number of neurons")
                    plt.savefig(f"./plots/predictivity_distribution_hist_new/{dataset}Prompt{model}_layer_{i}.png")
                    plt.close()
            
            else:
                sorted = np.sort(predictivities.flatten())[::-1]
                
                if topn!=None:
                    percent = int(len(sorted)/100*topn)
                    if mode=="max_seed":
                        # take last topn percent of sorted
                        sorted = sorted[:percent]
                        save_path=f"./plots/predictivity_distribution_max_seed/{seed}/topn_{topn}"

                    elif mode == "max_mean":
                        sorted = sorted[:percent]
                        sorted_list.append(sorted)
                        save_path=f"./plots/predictivity_distribution_{save_type}_max_mean/{seed}/topn_{topn}"

                    elif mode == "max_mean_location":
                        skill_neurons = skill_neurons[:percent]
                        sorted = np.zeros((12))
                        for i in range(12):
                            for j, neuron in enumerate(skill_neurons):
                                layer_idx, neuron_idx = neuron
                                if layer_idx == i:
                                    sorted[i] += 1

                        save_path=f"./plots/{save_type}_predictivity_distribution_max_mean_location/topn_{topn}"


                        sorted_list.append(sorted)

                    elif mode=="max_mean_100":
                        sorted = np.zeros((5,percent,100))
                        skill_neurons = skill_neurons[:percent]
                        for i, neuron in enumerate(skill_neurons):
                            layer_idx, neuron_idx = neuron
                            for j, raw_preds in enumerate(predictivities_raw):
                                if model == "T5":
                                    raw_preds = raw_preds[0]
                                sorted[j,i] = raw_preds[layer_idx, :100, neuron_idx]
                        if seed!="mean":
                            temp = seed_dict[str(seed)]
                            sorted = np.sort(sorted[temp].flatten())
                            save_path=f"./plots/predictivity_distribution_{save_type}_max_mean_100/seed_{seed}/topn_{topn}"
                        else:
                            sorted = np.sort(np.mean(sorted, axis=0).flatten())
                            save_path=f"./plots/predictivity_distribution_{save_type}_max_mean_100/mean/topn_{topn}"
                        sorted_list.append(sorted)

                    elif mode=="max_mean_230":
                        if model == "T5":
                            sorted = np.zeros((5,percent,230))
                        else:
                            sorted = np.zeros((5,percent,231))
                        skill_neurons = skill_neurons[:percent]
                        for i, neuron in enumerate(skill_neurons):
                            layer_idx, neuron_idx = neuron
                            for j, raw_preds in enumerate(predictivities_raw):
                                if model == "T5":
                                    raw_preds = raw_preds[0]
                                sorted[j,i] = raw_preds[layer_idx, :, neuron_idx]
                        if seed!="mean" and seed!=None:
                            temp = seed_dict[seed]
                            sorted = np.sort(sorted[temp].flatten())
                            save_path=f"./plots/predictivity_distribution_max_mean_230/seed_{seed}/topn_{topn}"
                        elif seed=="mean":
                            sorted = np.sort(np.mean(sorted, axis=0).flatten())
                            save_path=f"./plots/predictivity_distribution_max_mean_230/mean/topn_{topn}"
                        else:
                            sorted = np.sort(np.mean(sorted, axis=0).flatten())
                            save_path=f"./plots/predictivity_distribution_max_mean_230/mean/topn_{topn}"
                        

                    elif mode=="max_mean_5":
                        sorted = np.zeros((5,percent,1))
                        skill_neurons = skill_neurons[:percent]
                        for i, neuron in enumerate(skill_neurons):
                            layer_idx, neuron_idx = neuron
                            for j, raw_preds in enumerate(predictivities_raw):
                                if model == "T5":
                                    raw_preds = raw_preds[0]
                                max_tokens = np.max(raw_preds, axis=1)
                                max_token = max_tokens[layer_idx, neuron_idx]
                
                                sorted[j,i,0] = max_token

                        if seed!=None:
                            sorted = np.sort(sorted[seed].flatten())
                            save_path=f"/Users/leonackermann/Desktop/continuous_prompt_analysis/skill_neurons/plots/predictivity_distribution_max_mean_5/seed_{seed}/topn_{topn}"

                        else:
                            sorted = np.sort(np.mean(sorted, axis=0).flatten())
                            save_path=f"/Users/leonackermann/Desktop/continuous_prompt_analysis/skill_neurons/plots/predictivity_distribution_max_mean_5/mean/topn_{topn}"
                    elif mode == "nonadversarial_adversarial":
                        sorted_list.append(sorted)
                        save_path=f"./plots/predictivity_distribution_max_mean_adversarial_nonadversarial/{seed}/topn_{topn}"

                elif topn!= None and adversarial==True:
                    save_path=f"./plots/predictivity_distribution_max_mean_adversarial_nonadversarial/{seed}/topn_{topn}"
                else:
                    save_path=f"plots/predictivity_distribution_hist_new/"

                #fig, ax = plt.subplots(figsize=(6,4))
                #bin_boundaries = get_bin_boundaries(sorted)
                #print(model)
                #print(dataset)
                #print(bin_boundaries)
                #ax.hist(sorted,bins=bin_boundaries, edgecolor="black", linewidth=1.2)
                ##ax.set_title(f"Distribution of predictivities for {dataset}Prompt{model}")
                #ax.set_xlabel("Predictivity", fontsize=20)
                #ax.set_ylabel("Number of neurons", fontsize=20)
                #ax.set_xticks(bin_boundaries)
                #ax.tick_params(axis='x', labelsize=18)  # Adjust the labelsize as needed
                #ax.tick_params(axis='y', labelsize=18)  # Adjust the labelsize as needed        
                #plt.yscale("log")
                #for axis in [ax.yaxis]:
                #    axis.set_major_formatter(ScalarFormatter())
                #
                #plt.tight_layout()
#
                #try:
                #    os.mkdir(save_path)
                #except:
                #    pass
                #plt.savefig(f"{save_path}/{dataset}Prompt{model}.png")
                #plt.close()
        if location == False:
            fig, ax = plt.subplots(figsize=(6,5))
            bin_boundaries_list = [get_bin_boundaries(sorted) for sorted in sorted_list]
            maximum = np.max([np.max(bin_boundaries_list[0]), np.max(bin_boundaries_list[1])])
            minimum = np.min([np.min(bin_boundaries_list[0]), np.min(bin_boundaries_list[1])])
            bin_boundaries = np.arange(minimum, maximum, 0.05)
            print(model)
            print(dataset)
            print(bin_boundaries)
            ax.hist([sorted_list[0], sorted_list[1]],bins=bin_boundaries, edgecolor="black", linewidth=1.2, label=["roberta", "t5"])
            #ax.hist(,bins=bin_boundaries, edgecolor="black", linewidth=1.2, alpha=0.5, label="t5")
            #ax.set_title(f"Distribution of predictivities for {dataset}Prompt{model}")
            ax.set_xlabel("Predictivity", fontsize=20)
            ax.set_ylabel("Number of neurons", fontsize=20)
            ax.set_xticks(bin_boundaries)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.tick_params(axis='x', labelsize=18)  # Adjust the labelsize as needed
            ax.tick_params(axis='y', labelsize=18)  # Adjust the labelsize as needed        
            ax.legend(fontsize=18)
            plt.yscale("log")
            for axis in [ax.yaxis]:
                axis.set_major_formatter(ScalarFormatter())

            plt.tight_layout()
            try:
                os.mkdir(save_path)
            except:
                pass
            plt.savefig(f"{save_path}/{dataset}Prompt.png")
            plt.close()
        else:
            x_labels = range(1,13,1)
            x = np.arange(len(x_labels))
            width = 0.35

            fig, ax = plt.subplots()
            bars1 = ax.bar(x - width/2, sorted_list[0], width, label='Roberta ')
            bars2 = ax.bar(x + width/2, sorted_list[1], width, label='T5')

            ax.set_xlabel('Layer')
            ax.set_ylabel('Counts')
            ax.set_title('Bar Plot of Data Sets')
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            ax.legend()

            plt.tight_layout()
            try:
                os.mkdir(save_path)
            except:
                pass
            plt.savefig(f'{save_path}/{dataset}.png')
            plt.close()
            #fig, ax = plt.subplots(figsize=(6,4))
            #bin_boundaries = get_bin_boundaries(sorted)
            #print(model)
            #print(dataset)
            #print(bin_boundaries)
            #ax.hist(sorted,bins=bin_boundaries, edgecolor="black", linewidth=1.2)
            ##ax.set_title(f"Distribution of predictivities for {dataset}Prompt{model}")
            #ax.set_xlabel("Predictivity", fontsize=20)
            #ax.set_ylabel("Number of neurons", fontsize=20)
            #ax.set_xticks(bin_boundaries)
            #ax.tick_params(axis='x', labelsize=10)  # Adjust the labelsize as needed
            #ax.tick_params(axis='y', labelsize=18)  # Adjust the labelsize as needed        
            #plt.yscale("log")
            #for axis in [ax.yaxis]:
            #    axis.set_major_formatter(ScalarFormatter())
            #
            #plt.tight_layout()
            #try:
            #    os.mkdir(save_path)
            #except:
            #    pass
            #plt.savefig(f"{save_path}/{dataset}Prompt{model}.png")
            #plt.close()

def get_barplot_number_inhibitory_neuron_accuracies(datasets, models, seeds, inhibitory=True):

    count = np.zeros((len(models), len(datasets)))
    layer_count = np.zeros((len(models), len(datasets), 12))
    predictivities_inhibitory = np.zeros((len(models), len(datasets), len(seeds), 12, 100, 3072))
    quartiles_inhibitory = np.zeros((len(models), len(datasets), 12, 5))
    for i in range(len(models)):
        for j in range(len(datasets)):
            accuracies = np.array([np.load(f"../../../../../../Volumes/Extreme SSD/Projects/20230501_bachelor_thesis/neuron_accuracies_raw/{datasets[j]}Prompt{models[i]}_{seed}.npy") for seed in seeds]) 
            predictivities = np.array([np.load(f"../../../../../../Volumes/Extreme SSD/Projects/20230501_bachelor_thesis/neuron_predictivities_raw/{datasets[j]}Prompt{models[i]}_{seed}.npy") for seed in seeds])
            quartiles_temp = np.zeros((len(seeds), 12, 5))

            if models[i] == "T5":
                predictivities = predictivities[:,0]
                accuracies = accuracies[:,0]
            sums = np.zeros((len(seeds)))
            layer_sums = np.zeros((len(seeds), 12))
            for k in range(len(seeds)):
                accuracy = accuracies[k]    
                predictivity = predictivities[k]
                if inhibitory==True:
                    indices_inhibitory = np.where((1-accuracy)>accuracy, 1, 0)  
                else:
                    indices_inhibitory = np.where((1-accuracy)<accuracy, 1, 0)
                indices_max = np.argmax(predictivity, axis=1)

                # Create a new array with 1 at the index of the maximum and 0 elsewhere
                result = np.zeros_like(predictivity)
                result[np.arange(predictivity.shape[0])[:, np.newaxis], indices_max, np.arange(predictivity.shape[2])] = 1        

                sum = 0
                
                for a in range(12):
                    for b in range(100):
                        for c in range(3072):
                            if result[a][b][c] == 1 and indices_inhibitory[a][b][c] == 1:
                                sum += 1
                                layer_sums[k][a] += 1
                                predictivities_inhibitory[i][j][k][a][b][c] = accuracy[a][b][c]
                
                sums[k] = sum
                data =  predictivities_inhibitory[i][j][k]
                non_zero_mask = data!= 0
                for a in range(12):
                    if True in non_zero_mask[a]:
                        quartiles_temp[k][a] = np.percentile(data[a][non_zero_mask[a]], [0,0.25, 0.5, 0.75,1])
                    else:
                        quartiles_temp[k][a] = np.zeros((5))
                #filtered_data = data[non_zero_mask].reshape(12, -1)
                #filtered_data = data[non_zero_mask]

                #quartiles_temp[k] = np.percentile(filtered_data, [0.25, 0.5, 0.75])
            
            count[i,j] = np.mean(sums)
            quartiles_inhibitory[i][j] = np.mean(quartiles_temp, axis=0)
            
            
            layer_count[i][j] = np.mean(layer_sums, axis=0)
            #predictivities_inhibitory[i][j] = np.mean(predictivity_inhibitory, axis=0)
    maximum = np.max(accuracies)
    minimum = np.min(accuracies)

    
    #fig, ax = plt.subplots(figsize=(6,5))
    #plt.hist([accuracies[0], accuracies[1]], bins=len(datasets), alpha=0.7, label=models)
#
    ## Customize the plot
    #plt.title('Histogram of Values Below 0.5')
    #plt.xlabel('Number of Values Below 0.5')
    #plt.ylabel('Frequency')
    ## We want to show all ticks...
    #ax.set_xticks(np.arange(len(datasets)))
    #ax.set_yticks(np.arange(0, maximum, 5000))
    ## ... and label them with the respective list entries
    #ax.set_xticklabels(datasets)
    #ax.legend()
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
    ## Show the plot
    #plt.show()

    return count, layer_count, quartiles_inhibitory

def get_boxplot_neuron_pred_acc(datasets, models, mean=False, mode="predictivity", adversarial=False):
    """Create boxplots for neuron predictivities per layer of all datasets for all models, y-axis: performance [0,1], x-axis: layer [1,12]
    
    args:
        datasets: list of datasets
        models: list of models
        mean: 
            - if True, create boxplot for mean of neuron predictivities per layer across all datasets for each model, generates len(models) plots
            - if False, create boxplot for mean neuron predictivity per layer for each model and dataset, generates len(models)*len(datasets) plots
        mode: "predictivity" or "accuracy"
        adversarial: if True, change save_path of plots
    
    returns:
        None
    """
    predictivities = np.zeros((len(models), len(datasets), 12, 3072))
    if mode=="predictivity":
        path=f"./plots/boxplots_neuron_pred_per_layer/"
    else:
        path=f"./plots/boxplots_neuron_acc_per_layer/"

    try:
        os.mkdir(path)
    except:
        pass

    print("path", path)
    for i,model in enumerate(models):
        for j,dataset in enumerate(datasets):
            if mode=="predictivity":
                data = np.load(f"./skill_neuron_predictivities/{dataset}Prompt{model}_mean.npy")
            else:
                data = np.load(f"./skill_neuron_accuracies/{dataset}Prompt{model}_mean.npy")
            if mean==False:
                plt.figure(figsize=(7,5))
                plt.figure()  # Create a new figure for each dataset
                plt.boxplot(data.T, showfliers='diamond', notch=True, patch_artist=True, flierprops={'markersize': 2})  # Customize outliers
                plt.xlabel('Layer')
                plt.ylabel(f'Neuron {mode}')
                #plt.title(f'Predictivity for {dataset}{model}')
                # Save the plot to a file
                plt.savefig(f"{path}/{dataset}_{model}_boxplot.png", dpi=600)
                # Close the figure to free up memory
                plt.close()
            if mean==True:
                predictivities[i][j] = data
        if mean==True:
            plt.figure(figsize=(7,5))
            plt.figure()  # Create a new figure for each dataset
            #print("shape predictivities", predictivities[i].shape)
            data = np.mean(predictivities[i], axis=0)
            #print("shape data", data.shape)
            plt.boxplot(data.T, showfliers='diamond', notch=True, patch_artist=True, flierprops={'markersize': 2})  # Customize outliers
            plt.xlabel('Layer', fontsize=20)
            plt.ylabel(f'Neuron {mode}', fontsize=20)
            plt.xticks(fontsize=16)  # Increase the x tick label font size
            plt.yticks(fontsize=16)  # Increase the y tick label font size
            #plt.title(f'Mean of neuron predictivity across all datasets for {model}')
            # Save the plot to a file
            plt.tight_layout() 
            if adversarial==True:
                adversarial = "_adversarial"
                print("model")
                print("adversarial")
            else:
                adversarial = ""
            plt.savefig(f"{path}/{model}_boxplot_mean{adversarial}.png", dpi=600)
            plt.close()


def get_boxplot_barplot_neuron_model(datasets, models, seeds, mode="predictivity"):
    """Create boxplots for neuron predictivities/accuracies per dataset with model performance as barplot, y-axis: performance [0,1], x-axis: datasets

    args:
        datasets: list of datasets
        models: list of models
        seeds: list of seeds
        mode: "predictivity" or "accuracy"
    
    returns:
        None
    """

    predictivities = np.zeros((len(models), len(datasets), 12, 3072))
    accuracies_raw = np.zeros((len(models), len(datasets), 12, 1536000))
    if mode=="predictivity":
        load_path=f"./skill_neuron_predictivities/"
        save_path=f"./plots/boxplot_barplot_neuron_pred_model_acc"
    elif mode=="accuracy":
        load_path=f"./neuron_accuracies/"
        save_path=f"./plots/boxplot_barplot_neuron_acc_model_acc"
    elif mode=="accuracies_raw":
        load_path=f"../../../../../../Volumes/Extreme SSD/Projects/20230501_bachelor_thesis/neuron_accuracies_raw/"
        save_path=f"./plots/boxplot_barplot_neuron_acc_model_acc_new"

    try:
        os.mkdir(save_path)
    except:
        pass

    if mode=="predictivity":
        for i,model in enumerate(models):
            for j,dataset in enumerate(datasets):
                predictivities[i][j] = np.load(f"{load_path}/{dataset}Prompt{model}_mean.npy")

            # plt.figure()  # Create a new figure for each dataset
            fig, ax = plt.subplots()
            #max_predictivities = np.max(predictivities[i], axis=(1,2))
            data = predictivities[i].reshape((len(datasets),-1))
            #print("shape data", data.shape)
            #plt.boxplot(data.T, showfliers=True, )
            plt.boxplot(data.T, showfliers='diamond', zorder=2,notch=True, patch_artist=True, flierprops={'markersize': 2})  # Customize outliers

            plt.xlabel('Dataset', fontsize=14)
            plt.xticks(range(1, len(datasets)+1), datasets, rotation=45, ha='right', fontsize=12)
            #plt.yticks(range(0, 1))
            plt.ylabel(f'Neuron {mode}', fontsize=14)
            #plt.legend()
            plt.ylim(0,1)
            # Adding a line graph
            plt.twinx()  # Create a twin Axes sharing the xaxis

            _, mean_line = get_results(datasets, [model],seeds)
            mean_line = np.mean(mean_line, axis=2)[:,0]
            #print("mean line", mean_line.shape)
            plt.bar(range(1,len(datasets)+1),mean_line, color='green',label=f'Model accuracy', alpha=0.25, zorder=1, edgecolor='black', linewidth=1.2)
            plt.ylabel('Model Accuracy', fontsize=14)
            plt.xticks(range(1, len(datasets)+1), datasets,rotation=45, ha='right')
            tick_colors = ["midnightblue", "midnightblue", "darkmagenta", "darkmagenta", "darkmagenta", "blueviolet","blueviolet","blueviolet", "blueviolet", "darkred", "darkred", "darkred", "darkred"]
            #tick_colors = ["midnightblue", "midnightblue", "darkmagenta", "darkmagenta", "blueviolet", "blueviolet"]
            for i, ticklabel in enumerate(ax.get_xticklabels()):
                ticklabel.set_color(tick_colors[i])
            plt.legend()
            #plt.title(f'Skill Neuron {mode} and model accuracy of {model}')
            # Save the plot to a file
            plt.tight_layout()
            plt.savefig(f"{save_path}/{model}.png", dpi=600)

            plt.close()
    elif mode=="accuracies_raw":
        for i,model in enumerate(models):
            for j,dataset in enumerate(datasets):
                count = np.zeros((len(seeds), 12,100,3072))
                for k, seed in enumerate(seeds):
                    if model == "T5":
                        count[k] = np.load(f"{load_path}/{dataset}Prompt{model}_{seed}.npy")[0]
                    else:
                        count[k] = np.load(f"{load_path}/{dataset}Prompt{model}_{seed}.npy")
                accuracies_raw[i][j] = count.reshape((12,-1))#np.load(f"{load_path}/{dataset}Prompt{model}_mean.npy")

            # plt.figure()  # Create a new figure for each dataset
            fig, ax = plt.subplots()
            #max_predictivities = np.max(predictivities[i], axis=(1,2))
            data = accuracies_raw[i].reshape((len(datasets),-1))
            #print("shape data", data.shape)
            #plt.boxplot(data.T, showfliers=True, )
            plt.boxplot(data.T, showfliers='diamond', zorder=2,notch=True, patch_artist=True, flierprops={'markersize': 2})  # Customize outliers

            plt.xlabel('Dataset', fontsize=14)
            plt.xticks(range(1, len(datasets)+1), datasets, rotation=45, ha='right', fontsize=12)
            #plt.yticks(range(0, 1))
            plt.ylabel(f'Neuron Accuracy', fontsize=14)
            #plt.legend()
            plt.ylim(0,1)
            # Adding a line graph
            plt.twinx()  # Create a twin Axes sharing the xaxis

            _, mean_line = get_results(datasets, [model],seeds)
            mean_line = np.mean(mean_line, axis=2)[:,0]
            #print("mean line", mean_line.shape)
            plt.bar(range(1,len(datasets)+1),mean_line, color='green',label=f'Model accuracy', alpha=0.25, zorder=1, edgecolor='black', linewidth=1.2)
            plt.ylabel('Model Accuracy', fontsize=14)
            plt.xticks(range(1, len(datasets)+1), datasets,rotation=45, ha='right')
            tick_colors = ["midnightblue", "midnightblue", "darkmagenta", "darkmagenta", "darkmagenta", "blueviolet","blueviolet","blueviolet", "blueviolet", "darkred", "darkred", "darkred", "darkred"]
            #tick_colors = ["midnightblue", "midnightblue", "darkmagenta", "darkmagenta", "blueviolet", "blueviolet"]
            for i, ticklabel in enumerate(ax.get_xticklabels()):
                ticklabel.set_color(tick_colors[i])
            plt.legend()
            #plt.title(f'Skill Neuron {mode} and model accuracy of {model}')
            # Save the plot to a file
            plt.tight_layout()
            plt.savefig(f"{save_path}/{model}.png", dpi=600)



def get_lineplot_perturbation(datasets, models, seeds, type, mode):
    """Create lineplot with errorbars for perturbation accuracy of skill neurons and random neurons, y-axis: accuracy [0,1], x-axis: perturbation rate [2,4,6,...,30]
    
    args:
        datasets: list of datasets
        models: list of models
        seeds: list of seeds
        type: [skill_neuron_perturbation_type, random_neuron_perturbation_type]

    returns:
        None
    """

    perturbation_acc_skill_neurons = get_perturbation_accuracy_runs(datasets, models, 1, type[0], seeds, mode, layer=True)
    perturbation_acc_random_neurons = get_perturbation_accuracy_runs(datasets, models, 1, type[1], seeds, mode, layer=True)
    print(perturbation_acc_skill_neurons)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
    # iterate over dictionary
    for dataset_key, dataset_value in perturbation_acc_skill_neurons.items():
        for model_key, model_value in dataset_value.items():
            #x = np.arange(0,0.9,0.2) + 0.2
            x = [0,1,3,5,7,9,11,13,15]
            fig, ax = plt.subplots()
            mean_gaussian = np.array(perturbation_acc_skill_neurons[dataset_key][model_key]).T[0]
            sem_gaussian = np.array(perturbation_acc_skill_neurons[dataset_key][model_key]).T[1]
            #print(sem_gaussian)
            #sem_gaussian = np.arange(0,15,1)
            upper_bound_gaussian = mean_gaussian + sem_gaussian
            lower_bound_gaussian = mean_gaussian - sem_gaussian
            mean_random = np.array(perturbation_acc_random_neurons[dataset_key][model_key]).T[0]
            sem_random = np.array(perturbation_acc_random_neurons[dataset_key][model_key]).T[1]
            upper_bound_random = mean_random + sem_random
            lower_bound_random = mean_random - sem_random
            line_gaussian, = ax.plot(x, mean_gaussian, label="Skill neurons", color="tab:green", linewidth=3, linestyle="-")
            line_random, = ax.plot(x, mean_random, label="Random neurons", color="tab:brown", linewidth=3, linestyle="-")
            # Plot error bars

            plt.errorbar(x, mean_gaussian, yerr=sem_gaussian, fmt='none', capsize=4, ecolor='tab:green', linewidth=1.5)
            #plt.fill_between(x, upper_bound_gaussian, lower_bound_gaussian, alpha=0.2)

            plt.errorbar(x, mean_random, yerr=sem_random, fmt='none', capsize=4, ecolor='tab:brown', linewidth=1.5)
            #plt.fill_between(x, upper_bound_random, lower_bound_random, alpha=0.2)

            # Fill between the upper and lower bounds
            #plt.fill_between(x, lower_bound, upper_bound, alpha=0.2)
            ax.set_xlabel("Perturbation rate (%)", fontsize=16)
            ax.set_xticks(x)  # Set x-axis ticks to the desired positions
            ax.set_ylabel("Model Accuracy", fontsize=16)
            #ax.set_title(f"{model_key} on {dataset_key}")
            ax.tick_params(axis='x', labelsize=14)  # Adjust the labelsize as needed
            ax.tick_params(axis='y', labelsize=14)  # Adjust the labelsize as needed        
            ax.legend(fontsize=14)
            #plt.show()
            fig.tight_layout()
            save_path = f"suppression_layer_{mode}_1-15"
            try:
                os.mkdir(save_path)
            except:
                pass
            fig.savefig(f"./plots/{save_path}/{dataset_key}Prompt{model_key}.png")
            plt.close()


def get_lineplots_perturbation(datasets, models, seeds, type, mode, adv_nonAdv = "_sn"):
    """Create lineplot with errorbars for perturbation accuracy of skill neurons and random neurons, y-axis: accuracy [0,1], x-axis: perturbation rate [2,4,6,...,30]
    
    args:
        datasets: list of datasets
        models: list of models
        seeds: list of seeds
        type: [skill_neuron_perturbation_type, random_neuron_perturbation_type]

    returns:
        None
    """

    perturbation_acc_skill_neurons = get_perturbation_accuracy_runs(datasets, models, 1, type[0], seeds, mode, layer=True, adv_nonAdv=adv_nonAdv)
    perturbation_acc_random_neurons = get_perturbation_accuracy_runs(datasets, models, 1, type[1], seeds, mode, layer=True, adv_nonAdv=adv_nonAdv)
    print(perturbation_acc_skill_neurons)
    colors = ["tab:orange", "tab:blue"]#, "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
    color_group = [['#1f77b4', '#aec7e8'], ['#ff7f0e', '#ffbb78']]  # Blue shades
    color_group2 = ['#ff7f0e', '#ffbb78']  # Orange shades
    # iterate over dictionary
    #for model in models:
    
    for dataset_key, dataset_value in perturbation_acc_skill_neurons.items():
        x = [0,1,3,5,7,9,11,13,15]

        fig, ax = plt.subplots(figsize=(8,5))
        i = 0
        
        for model_key, model_value in dataset_value.items():
            #if model==model_key:
            
            mean_gaussian = np.array(perturbation_acc_skill_neurons[dataset_key][model_key]).T[0]
            sem_gaussian = np.array(perturbation_acc_skill_neurons[dataset_key][model_key]).T[1]
            #print(sem_gaussian)
            #sem_gaussian = np.arange(0,15,1)
            upper_bound_gaussian = mean_gaussian + sem_gaussian
            lower_bound_gaussian = mean_gaussian - sem_gaussian
            mean_random = np.array(perturbation_acc_random_neurons[dataset_key][model_key]).T[0]
            sem_random = np.array(perturbation_acc_random_neurons[dataset_key][model_key]).T[1]
            upper_bound_random = mean_random + sem_random
            lower_bound_random = mean_random - sem_random
            line_gaussian, = ax.plot(x, mean_gaussian, label=f"{model_key} skill", color=color_group[i][0], linewidth=3, linestyle="-")
            line_random, = ax.plot(x, mean_random, label=f"{model_key} random", color=color_group[i][0], linewidth=3, linestyle="--")
            # Plot error bars
            eb1=plt.errorbar(x, mean_random, yerr=sem_random, capsize=4, ecolor=color_group[i][0], linewidth=2, linestyle="dashed", fmt="None")
            eb1[-1][0].set_linestyle('--')
            plt.errorbar(x, mean_gaussian, yerr=sem_gaussian, capsize=4, ecolor=color_group[i][0], linewidth=2, linestyle="-", fmt="None")
            #plt.fill_between(x, upper_bound_gaussian, lower_bound_gaussian, alpha=0.2)
            
            #plt.fill_between(x, upper_bound_random, lower_bound_random, alpha=0.2)
            i += 1
        # Fill between the upper and lower bounds
        #plt.fill_between(x, lower_bound, upper_bound, alpha=0.2)
        ax.set_xlabel("Suppression rate (%)", fontsize=24)
        ax.set_xticks(x)  # Set x-axis ticks to the desired positions
        ax.set_ylabel("Model Accuracy", fontsize=24)
        #ax.set_title(f"{model_key} on {dataset_key}")
        ax.tick_params(axis='x', labelsize=20)  # Adjust the labelsize as needed
        ax.tick_params(axis='y', labelsize=20)  # Adjust the labelsize as needed
        #ax.set_legend(datasets)      
        #legend_patches = [[color, desc] for color, desc in zip(colors, datasets)]  
        ax.legend(fontsize=18)        #plt.show()
        fig.tight_layout()
        save_path = f"suppression_analysis_adv_nonAdv"
        try:
            os.mkdir(save_path)
        except:
            pass
        fig.savefig(f"./plots/{save_path}/{dataset_key}.png")
        plt.close()


def get_lineplot_correlation_per_topn(datasets, models, topn):
    """Create lineplot with errorbars for perturbation accuracy of skill neurons and random neurons, y-axis: accuracy [0,1], x-axis: perturbation rate [2,4,6,...,30]
    
    args:
        datasets: list of datasets
        models: list of models
        seeds: list of seeds
        type: [skill_neuron_perturbation_type, random_neuron_perturbation_type]

    returns:
        None
    """

    topn_correlations = get_correlation_per_topn(datasets, models, topn)
    # iterate over dictionary
    for dataset_key, dataset_value in topn_correlations.items():
        for model_key, model_value in dataset_value.items():
            x = np.arange(2,topn+1,2)

            fig, ax = plt.subplots()
            mean_gaussian = np.array(topn_correlations[dataset_key][model_key]).T[0]
            sem_gaussian = np.array(topn_correlations[dataset_key][model_key]).T[1]
            #print(sem_gaussian)
            #sem_gaussian = np.arange(0,15,1)
            upper_bound_gaussian = mean_gaussian + sem_gaussian
            lower_bound_gaussian = mean_gaussian - sem_gaussian
            #mean_random = np.array(perturbation_acc_random_neurons[dataset_key][model_key]).T[0]
            #sem_random = np.array(perturbation_acc_random_neurons[dataset_key][model_key]).T[1]
            #upper_bound_random = mean_random + sem_random
            #lower_bound_random = mean_random - sem_random
            line_gaussian, = ax.plot(x, mean_gaussian, label="Skill neurons", color="tab:green", linewidth=3)
            #line_random, = ax.plot(x, mean_random, label="Random neurons", color="tab:brown", linewidth=3)
            # Plot error bars

            plt.errorbar(x, mean_gaussian, yerr=sem_gaussian, fmt='none', capsize=4, ecolor='tab:green', linewidth=1.5)
            #plt.fill_between(x, upper_bound_gaussian, lower_bound_gaussian, alpha=0.2)

            #plt.errorbar(x, mean_random, yerr=sem_random, fmt='none', capsize=4, ecolor='tab:brown', linewidth=1.5)
            #plt.fill_between(x, upper_bound_random, lower_bound_random, alpha=0.2)

            # Fill between the upper and lower bounds
            #plt.fill_between(x, lower_bound, upper_bound, alpha=0.2)
            ax.set_xlabel("Topn neurons (%)", fontsize=20)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
            ax.set_xticks(x)  # Set x-axis ticks to the desired positions
            ax.set_ylabel("Correlation between seeds", fontsize=20)
            #ax.set_title(f"{model_key} on {dataset_key}")
            ax.tick_params(axis='x', labelsize=5)  # Adjust the labelsize as needed
            ax.tick_params(axis='y', labelsize=18)  # Adjust the labelsize as needed        
            #ax.legend(fontsize=14)


            #plt.show()
            fig.tight_layout()
            save_path = f"correlation_lineplot"
            try:
                os.mkdir(save_path)
            except:
                pass
            fig.savefig(f"./plots/{save_path}/{dataset_key}Prompt{model_key}.png")
            plt.close()

def get_lineplot_dataset_correlation_per_topn(models, topn):
    """Create lineplot with errorbars for perturbation accuracy of skill neurons and random neurons, y-axis: accuracy [0,1], x-axis: perturbation rate [2,4,6,...,30]
    
    args:
        datasets: list of datasets
        models: list of models
        seeds: list of seeds
        type: [skill_neuron_perturbation_type, random_neuron_perturbation_type]

    returns:
        None
    """
    tasks = ["ethical_judgment", "paraphrase_identification", "sentiment_analysis", "average"]
    task_correlations = get_dataset_correlation_per_topn(tasks, models, topn)
    
    

    # iterate over dictionary
    for task_key, task_value in task_correlations.items():
        fig, ax = plt.subplots()
        for model_key, model_value in task_value.items():
            x = np.arange(2,topn+1,2)
            
            mean_gaussian = np.array(task_correlations[task_key][model_key]).T[0]
            sem_gaussian = np.array(task_correlations[task_key][model_key]).T[1]
            #print(sem_gaussian)
            #sem_gaussian = np.arange(0,15,1)
            upper_bound_gaussian = mean_gaussian + sem_gaussian
            lower_bound_gaussian = mean_gaussian - sem_gaussian
            #mean_random = np.array(perturbation_acc_random_neurons[dataset_key][model_key]).T[0]
            #sem_random = np.array(perturbation_acc_random_neurons[dataset_key][model_key]).T[1]
            #upper_bound_random = mean_random + sem_random
            #lower_bound_random = mean_random - sem_random
            
            #line_random, = ax.plot(x, mean_random, label="Random neurons", color="tab:brown", linewidth=3)
            # Plot error bars
            if model_key == "Roberta":
                color = "tab:blue"
            else:
                color = "tab:orange"
            line_gaussian, = ax.plot(x, mean_gaussian, color=color, label=model_key, linewidth=3)
            plt.errorbar(x, mean_gaussian, yerr=sem_gaussian,color=color, fmt='none', capsize=4, linewidth=1.5)
            

            #plt.fill_between(x, upper_bound_gaussian, lower_bound_gaussian, alpha=0.2)
            #plt.errorbar(x, mean_random, yerr=sem_random, fmt='none', capsize=4, ecolor='tab:brown', linewidth=1.5)
            #plt.fill_between(x, upper_bound_random, lower_bound_random, alpha=0.2)
            # Fill between the upper and lower bounds
            #plt.fill_between(x, lower_bound, upper_bound, alpha=0.2)
            ax.set_xlabel("Topn neurons (%)", fontsize=20)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
            ax.set_xticks(x)  # Set x-axis ticks to the desired positions
            ax.set_ylabel("Correlation between seeds", fontsize=20)
            #ax.set_title(f"{model_key} on {dataset_key}")
            ax.tick_params(axis='x', labelsize=5)  # Adjust the labelsize as needed
            ax.tick_params(axis='y', labelsize=18)  # Adjust the labelsize as needed        
            ax.legend(fontsize=14)


        #plt.show()
        fig.tight_layout()
        save_path = f"dataset_correlation_lineplot"
        try:
            os.mkdir(save_path)
        except:
            pass
        fig.savefig(f"./plots/{save_path}/{task_key}.png")
        plt.close()

def get_lineplot_overlap_nonadversarial_adversarial(datasets, models, topn):
    """Create lineplot with errorbars for perturbation accuracy of skill neurons and random neurons, y-axis: accuracy [0,1], x-axis: perturbation rate [2,4,6,...,30]
    
    args:
        datasets: list of datasets
        models: list of models
        seeds: list of seeds
        type: [skill_neuron_perturbation_type, random_neuron_perturbation_type]

    returns:
        None
    """
       
    overlaps = get_adv_overlap_per_topn(datasets, models, topn)
    
    # iterate over dictionary
    for dataset_key, dataset_value in overlaps.items():
        fig, ax = plt.subplots()
        for model_key, model_value in dataset_value.items():
            x = np.arange(2,topn+1,2)
            
            mean_gaussian = np.array(overlaps[dataset_key][model_key]).T[0]
            sem_gaussian = np.array(overlaps[dataset_key][model_key]).T[1]
            #print(sem_gaussian)
            #sem_gaussian = np.arange(0,15,1)
            upper_bound_gaussian = mean_gaussian + sem_gaussian
            lower_bound_gaussian = mean_gaussian - sem_gaussian
            #mean_random = np.array(perturbation_acc_random_neurons[dataset_key][model_key]).T[0]
            #sem_random = np.array(perturbation_acc_random_neurons[dataset_key][model_key]).T[1]
            #upper_bound_random = mean_random + sem_random
            #lower_bound_random = mean_random - sem_random
            
            #line_random, = ax.plot(x, mean_random, label="Random neurons", color="tab:brown", linewidth=3)
            # Plot error bars
            if model_key == "Roberta":
                color = "tab:blue"
            else:
                color = "tab:orange"
            line_gaussian, = ax.plot(x, mean_gaussian, color=color, label=model_key, linewidth=3)
            plt.errorbar(x, mean_gaussian, yerr=sem_gaussian,color=color, fmt='none', capsize=4, linewidth=1.5)
            

            #plt.fill_between(x, upper_bound_gaussian, lower_bound_gaussian, alpha=0.2)
            #plt.errorbar(x, mean_random, yerr=sem_random, fmt='none', capsize=4, ecolor='tab:brown', linewidth=1.5)
            #plt.fill_between(x, upper_bound_random, lower_bound_random, alpha=0.2)
            # Fill between the upper and lower bounds
            #plt.fill_between(x, lower_bound, upper_bound, alpha=0.2)
            ax.set_xlabel("Topn neurons (%)", fontsize=20)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
            ax.set_xticks(x)  # Set x-axis ticks to the desired positions
            ax.set_ylabel("Correlation between seeds", fontsize=20)
            #ax.set_title(f"{model_key} on {dataset_key}")
            ax.tick_params(axis='x', labelsize=5)  # Adjust the labelsize as needed
            ax.tick_params(axis='y', labelsize=18)  # Adjust the labelsize as needed        
            ax.legend(fontsize=14)


        #plt.show()
        fig.tight_layout()
        save_path = f"adv_overlap_lineplot"
        try:
            os.mkdir(save_path)
        except:
            pass
        fig.savefig(f"./plots/{save_path}/{dataset_key}.png")
        plt.close()
    

# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+')
    parser.add_argument('--models', type=str, nargs='+')
    parser.add_argument('--seeds', type=str, nargs='+')
    parser.add_argument('--plot_transferability', '-c', help="specific config file", default="False")
    parser.add_argument('--relative', default="True")
    parser.add_argument('--plot_neuron_specificity', default="False")
    parser.add_argument('--normalized', default="False")
    parser.add_arugment('--plot_comparison', default="False")
    parser.add_argument('--plot_suppression', default="False")
    args = parser.parse_args()

    if args.plot_transferability=="True":
        if args.relative == "True":
            relative = True
        else:
            relative = False
        for model in args.models:
            plot_transferability_matrix(args.datasets,model, args.seeds,relative=relative)

    if args.plot_neuron_specificity=="True":
        if args.normalized == "True":
            normalized = True
        else:
            normalized = False
        for model in args.models:
            plot_matrix(args.datasets, model,data="predictivity", mode="across_layer_all", title= "", seed="mean", topn=100, normalized=normalized)

    if args.plot_comparison=="True":
        get_boxplot_barplot_neuron_model(args.datasets, args.models, args.seeds, mode="predictivity")

    if args.plot_suppression=="True":
        types=["skill_neurons", "random_neurons"]
        get_lineplots_perturbation(args.datasets, args.models, args.seeds, types, "max_mean_230")