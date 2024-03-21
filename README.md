# On the Relationship between Skill Neurons and the Robustness in Prompt Tuning
This is the source code of the paper "On the Relationship between Skill Neurons and the Robustness in Prompt Tuning", published at COLING 2024 and NeurIPS 2023 workshop.

## Table of Contents
- [Overview](#overview)
- [Reproduce the results](#reproduce-the-results)
- [Contact and Paper Information](#contact-and-paper-information)
- [Citation](#citation)


## Overview 

Prompt Tuning is a popular parameter-efficient finetuning method for pre-trained large language models (PLMs). Recently, based on experiments with RoBERTa,it has been suggested that Prompt Tuning activates specific neurons in the transformer’s feed-forward networks, that are highly predictive and selective for the given task. In this paper, we study the robustness of Prompt Tuning in relation to these “skill neurons”, using RoBERTa and T5. We show that prompts tuned for a specific task are transferable to tasks of the same type but are not very robust to adversarial data, with higher robustness for T5 than RoBERTa. At the same time, we replicate the existence of skill neurons in RoBERTa and further show that skill neurons also seem to exist in T5. Interestingly, the skill neurons of T5 determined on non-adversarial data are also among the most predictive neurons on the adversarial data, which is not the case for RoBERTa. We conclude that higher adversarial robustness may be related to a model’s ability to activate the relevant skill neurons on adversarial data.

## Reproduce the results

### Requirements
Create a conda environment with the required packages using the following command. 
```bash
conda env create -f environment.yml
```

Then activate the environment using the following command. 
```bash
conda activate robust-neurons
```

### Tune Prompts
To tune the prompts used in the experiments, we refer to the repository of Su et al. and their paper on [Prompt Transferability](https://github.com/thunlp/Prompt-Transferability).

### Verify Prompt Transferability
Plot the prompt transferability of prompts trained on certain `datasets`, one `model` and multiple `seeds` using the following command:
```bash
python src/analysis.py --plot_transferability True \
                    --datasets ${dataset1} ${dataset2} ... \
                    --model ${model} \
                    --seeds ${seed1} ${seed2} ...
```

### Run the experiments

#### Compute Baseline activations
Calculate the average activations of all neurons in each feed-forwrard network for a certain `dataset`,`model` and prompt trained on a certain `seed` using the following command. Do this for the adversarial and the non-adversarial dataset by passing the `--adversarial` flag.
```bash
cd src
python test.py --config config/${dataset}Prompt${model}.config \
                --prompt_seed ${seed} \
                --compute_baseline True \
                --adversarial True or False
```


#### Compute Neuron accuracies and predictivities
Calculate the neuron accuracies and predictivities for a certain `dataset`,`model` and prompt trained on a certain `seed` using the following command. Do this for the adversarial and the non-adversarial dataset by passing the `--adversarial` flag:
```bash
cd src
python test.py --config config/${dataset}Prompt${model}.config \
                --prompt_seed ${seed} \
                --compute_accuracy True \
                --compute_predictivity True 
                --adversarial True or False
```
#### Compute skill neurons and random neurons
Create skill neurons of neuron predictivities for multiple datasets, models and seeds using the following command. Do this for the adversarial and the non-adversarial dataset by passing the `--adversarial` flag.
```bash
cd src
python find.py --compute_skill_neurons True \
                --datasets ${dataset1} ${dataset2} ...\
                --models ${model} \
                --prompt_seed ${seed1} ${seed2} ...
```
Here we only need to generate random neurons once for each dataset and model combination
```bash
cd src
python find.py --compute_random_neurons True \
                --datasets ${dataset1} ${dataset2} ...\
                --models ${model} \
```

#### Compute model accuracy with suppressed neurons
Calculate the model accuracy with suppressed neurons for a certain `dataset`,`model` and prompt trained on a certain `seed` and `n`% suppressed neurons. Depending on the mask, either random neurons or skill neurons are suppressed. :
```bash
cd src
python test.py --config config/${dataset}Prompt${model}.config \
                --prompt_seed ${seed} \
                --perturbation_analysis True \
                --topn ${n} \
                --masks "random_neurons" or "skill_neurons"
```



### Plot the results

#### Plot Neuron predictivities vs. model accuracy

```bash
python analysis.py --plot_comparison True \
                    --datasets ${dataset1} ${dataset2} ... \
                    --models ${model} \
                    --prompt_seeds ${seed1} ${seed2} ...
```

#### Plot Skill Neuron specificity

```bash
python analysis.py --plot_neuron_specificity True \
                    --datasets ${dataset1} ${dataset2} ... \
                    --models ${model} \
                    --prompt_seeds ${seed1} ${seed2} ...
```

#### Skill Neurons suppression

```bash
python analysis.py --plot_suppression True \
                    --datasets ${dataset1} ${dataset2} ... \
                    --models ${model} \
                    --prompt_seeds ${seed1} ${seed2} ...
```




## Contact and Paper Information
For any questions, please contact the authors of the paper:
- Xenia Ohmer: xenia.ohmer@uni-osnabrueck.de
- Leon Ackermann: lackermann@uni-osnabrueck.de

## Citation
Please cite our paper if you find it useful for your research:


