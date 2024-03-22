# On the relationship between skill neurons and robustness in Prompt Tuning

This repository accompanies the publication:

Leon Ackermann & Xenia Ohmer. 2024. On the Relationship between Skill Neurons and Robustness in Prompt Tuning. *The 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING) (accepted).*

## Table of Contents
- [Overview](#overview)
- [Reproducing the results](#reproducing-the-results)
- [Citation](#citation)
- [Contact](#contact)


## Overview 

Prompt Tuning is a popular parameter-efficient finetuning method for pre-trained large language models (PLMs).
Recently, based on experiments with RoBERTa, it has been suggested that Prompt Tuning activates specific neurons
in the transformer’s feed-forward networks, that are highly predictive and selective for the given task. In this paper,
we study the robustness of Prompt Tuning in relation to these “skill neurons”, using RoBERTa and T5. We show that
prompts tuned for a specific task are transferable to tasks of the same type but are not very robust to adversarial data.
While prompts tuned for RoBERTa yield below-chance performance on adversarial data, prompts tuned for T5 are
slightly more robust and retain above-chance performance in two out of three cases. At the same time, we replicate
the finding that skill neurons exist in RoBERTa and further show that skill neurons also exist in T5. Interestingly,
the skill neurons of T5 determined on non-adversarial data are also among the most predictive neurons on the
adversarial data, which is not the case for RoBERTa. We conclude that higher adversarial robustness may be related
to a model’s ability to consistently activate the relevant skill neurons on adversarial data.

## Reproducing the results

### Requirements
You can create a conda environment with the required packages using the following command:
```bash
conda env create -f environment.yml
```

### Prompt Tuning
For Prompt Tuning, we used the [repository](https://github.com/thunlp/Prompt-Transferability) by Su et al. accompanying their paper on [Prompt Transferability](https://aclanthology.org/2022.naacl-main.290/). You can find their and our tuned prompts in the folder *continuous_prompts/*.

### Prompt transferability
The prompt transferability of the prompts tuned for a specific `model` on certain `datasets` and with different `seeds` can be plotted with:
```bash
python src/analysis.py --plot_transferability True \
                    --datasets ${dataset1} ${dataset2} ... \
                    --model ${model} \
                    --seeds ${seed1} ${seed2} ...
```

### Skill neuron analyses

#### Computing the baseline activations
You can calculate the average activations of all neurons in each feed-forward network for a certain `dataset`, `model`, and `seed`, with the following command. The flag `--adversarial` can be used to distinguish between adversarial and non-adversarial data.
```bash
cd src
python test.py --config config/${dataset}Prompt${model}.config \
                --prompt_seed ${seed} \
                --compute_baseline True \
                --adversarial True or False
```


#### Computing neuron accuracies and predictivities
To calculate the neuron accuracies and predictivities for a certain `dataset`, `model`, and `seed`, use the following command. Again, the flag `--adversarial` can be used to choose between adversarial and non-adversarial data:
```bash
cd src
python test.py --config config/${dataset}Prompt${model}.config \
                --prompt_seed ${seed} \
                --compute_accuracy True \
                --compute_predictivity True 
                --adversarial True or False
```

#### Extracting skill neurons and random neurons
To determine the skill neurons from the neuron predictivities for a set of `datasets`, a `model`, and a specific `seed` you can use the following command, again with the option to select adversarial and non-adversarial data with the `--adversarial` flag. 
```bash
cd src
python find.py --compute_skill_neurons True \
                --datasets ${dataset1} ${dataset2} ...\
                --models ${model} \
                --prompt_seed ${seed1} ${seed2} ...
```

We select a set of random neurons once per dataset and model with:
```bash
cd src
python find.py --compute_random_neurons True \
                --datasets ${dataset1} ${dataset2} ...\
                --models ${model} \
```

#### Suppression analysis
The suppression analysis for a certain `dataset`, `model`, `seed`, and `n`% suppressed neurons can be run with the following command. Depending on the mask, either random neurons or skill neurons are suppressed.
```bash
cd src
python test.py --config config/${dataset}Prompt${model}.config \
                --prompt_seed ${seed} \
                --perturbation_analysis True \
                --topn ${n} \
                --masks "random_neurons" or "skill_neurons"
```



### Plots

#### Neuron predictivities and model accuracy

```bash
python analysis.py --plot_comparison True \
                    --datasets ${dataset1} ${dataset2} ... \
                    --models ${model} \
                    --prompt_seeds ${seed1} ${seed2} ...
```

#### Task-specificity

```bash
python analysis.py --plot_neuron_specificity True \
                    --datasets ${dataset1} ${dataset2} ... \
                    --models ${model} \
                    --prompt_seeds ${seed1} ${seed2} ...
```

#### Suppression analysis

```bash
python analysis.py --plot_suppression True \
                    --datasets ${dataset1} ${dataset2} ... \
                    --models ${model} \
                    --prompt_seeds ${seed1} ${seed2} ...
```


## Citation
Please cite our paper if it is helpful to your work!
(Bibtex entry will follow)

## Contact 
If you have any questions do not hesitate to contact the corresponding author at lackermann@uni-osnabrueck.de.





