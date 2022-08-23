# A Universal Adversarial Policy for Text Classifiers

This repository is the official implementation of LUNATC from "A Universal Adversarial Policy for Text Classifiers". https://doi.org/10.1016/j.neunet.2022.06.018 

![universal adversarial policy overview](https://github.com/gallilmaimon/LUNATC/blob/master/images/universal_comparison.png)

## Setup

1) create an environment:
- This git was tested using python 3.7 and cuda 10.2, other versions may require some adjustments.
- We recommend using a virtual environment.
- Install the required depenedencies based on the given [requirements.txt](https://github.com/gallilmaimon/LUNATC/blob/master/requirements.txt):
```
pip install -r requirements.txt
```
* Note: these are extended dependencies which might not be needed for all baselines and environemnts, you can also install requirements by use.

2) Download word vectors, Universal Sentence Encoder and parse, by running the following command. see more in [resources\README.md](https://github.com/gallilmaimon/LUNATC/blob/master/resources/README.md):
```
./resources/preprocess.sh
```

## Preparing data & training the attacked models
Download the wanted pre-trained text classifier, and corresponding dataset to the folder: data/{dataset_name}. Where {dataset_name} is the name in brackets () in the following table.
We report the accuracy of each model on the preprocessed test set, however the data samples provided will not perfectly match the accuracy because they are filtered for length and possibly only part of the classes (this is to make attack more relevant and efficient).

| Dataset\Classifier      | BERT                                | Word-LSTM                           |
| :-----------------------|:----------------------------------  | :-----------------------------------|
|                         | **Accuracy, model link, data link** | **Accuracy, model link, data link** |
| IMDB (aclImdb)          | 93.98, [model](https://drive.google.com/file/d/1MtEzBmLmSn4ad-EefalzBOYyZZzltq71/view?usp=sharing), [data](https://drive.google.com/file/d/1TrpBUjGJVkObN8ktaKR2XLCXU1QkyHay/view?usp=sharing)                  |  85.70, [model](https://drive.google.com/file/d/1VQnW5uDhFGgHHj1BB4Iazt5evA_EnSRL/view?usp=sharing), [data](https://drive.google.com/file/d/1Hg-M4xmfkZ_RYVkNI9J5mgdezS2WQjZI/view?usp=sharing)                 |
| Toxic-Wikipedia (toxic) | 91.80, [model](https://drive.google.com/file/d/18NvVtoQovvzY5VKepIPUBje6ApcT1rqD/view?usp=sharing), [data](https://drive.google.com/file/d/1B6TxIZPlA19tVwkq6aF4WtjUPP2ZjMNr/view?usp=sharing)                  |  92.77, [model](https://drive.google.com/file/d/14qvEc34t_1iMNoATBGRFPF3OFyLR7qIK/view?usp=sharing), [data](https://drive.google.com/file/d/16s5bSRklIWKGCuEeG1rzcS6LGIM17cYo/view?usp=sharing)                 |
| Pubmed (pubmed)         | 96.75, [model](https://drive.google.com/file/d/1jvSXDL_TXqNEJqesh-0DA8cYArcz8rLz/view?usp=sharing), [data](https://drive.google.com/file/d/1y79fZv8cnq15ITVAHdJH-ZSbk8xlqJoT/view?usp=sharing)                  |  95.71, [model](https://drive.google.com/file/d/1-6BwUbAe6Ovx4OPa9HWvO4v-HacWNlpt/view?usp=sharing), [data](https://drive.google.com/file/d/1H-4jQpQ2Ei66GiwA5ow9iQoJq4-YLSnp/view?usp=sharing)                 |
| MNLI (mnli)             | 83.72, [model](https://drive.google.com/file/d/176MzUES9ltMGGaEoWYC4_CVt_gF1_rac/view?usp=sharing), [data](https://drive.google.com/file/d/1iMA13MEFPkAOeYgtgbUEYEQkpjPo5TqG/view?usp=sharing)                  | -                                   |


* For training from scratch, switching classifiers, using new datasets, changing data preprocess or anything else like that, see the following readme [data/README.md](https://github.com/gallilmaimon/LUNATC/blob/master/data/README.md).

## Performing the attacks
We focus on the attacks described in the paper, though genetic attacks and other variants of existing attacks are also supported.
### Non-Universal Baselines
- **Textfooler**: In order to attack using our implementation of textfooler (with the similarity threshold of tf-adjusted), use the following script:
```
python src/Attacks/textfooler_attack.py
```
\* Note that, you can manually change the similarity threshold to match the original paper within the code.

- **PWWS (ours)**: In order to attack using our implementation of PWWS, which uses the same actions as Textfooler, but uses the PWWS heuristic to choose the action order, use the same script. __But first__, change within the file: attack_type = 'pwws'.
```
python src/Attacks/textfooler_attack.py
```

- **PWWS (original)**: In order to attack with the original pwws, run:
```
python src/Attacks/pwws_attack.py
```
\* Note that the use of Named Entity replacement action can be switched off from within the code.

- **Simple Search**: In order to attack using the simple search baseline, follow these commands (and manual updates to the dataset and indices):

\* __Note!__ - that this baseline is used to calculate indices, not as "Simple Search" in results! to reach that, change num_rounds in the config file to 1.
```
cp src/Config/constants_search.yml src/Config/constants.yml
# Manually update data path and indices to attack (in order to only attack those of interest or split for different compute nodes)
# Run the attack
python src/Attacks/DQN_attack.py
```

### GenFooler
This universal baseline is also availble to run. First, change the parameters within the main function to point to your wished dataset, model type and attack type generalising ('tf' or 'pwws'). The default parameters match those used to get the results in the paper. Then run:
```
python src/Attacks/genfooler_attack.py
```

### LUNATC
To attack using LUNATC described in the paper, first train the agent by copying contents of [constants_lunatc_train.yml](https://github.com/gallilmaimon/LUNATC/blob/master/src/Config/constants_lunatc_train.yml) to [constants.yml](https://github.com/gallilmaimon/LUNATC/blob/master/src/Config/constants.yml). Afterwards update the dataset (indicated by the base path), the indices (based on those clculated in the previous section). Finally, update the parameters in the config according to the following table. Seeds used are 42, 43, 44. Model type depends on the attacked model. Other fields should remain the same.

| Parameter\Train Size | 500       | 750   | 950   | 25k   | 50k   |
| :--------------------|:----------| :-----| :-----| :-----| :-----|
| Num Episodes         | 25000     | 25000 | 25000 | 50000 | 50000 |
| Memory Size          | 10000     | 15000 | 20000 | 25000 | 25000 |
| Target Update        | 2500      | 3750  | 5000  | 12500 | 12500 |
| Eps Decay            | 7500      | 11250 | 15000 | 37500 | 37500 |
| Num Rounds (in code) | 2         |    4  |  4    | 3     | 1     |


Afterwards run this command, in order to train:
```LUNATC universal attack
python src/Attacks/DQN_attack.py 
```

After that perform the attack by inferncing with the trained model - copy contents of [constants_lunatc_test.yml](https://github.com/gallilmaimon/LUNATC/blob/master/src/Config/constants_lunatc_test.yml) to [constants.yml](https://github.com/gallilmaimon/LUNATC/blob/master/src/Config/constants.yml), update data  path and indices. Finally, run: 
```LUNATC universal attack
python src/Attacks/DQN_attack.py 
```


## Evaluation

To evaluate the baselines' attack success rate and oracle access, use [Analysis.ipynb](https://github.com/gallilmaimon/LUNATC/blob/master/Analysis.ipynb). It can be used to calculate the success rate and oracle access of all the models in the paper, thus allowing to recreate all results from the paper.

## Citation
If you found this work useful, please cite the following related article:

```
@article{MAIMON2022282,
title = {A universal adversarial policy for text classifiers},
journal = {Neural Networks},
volume = {153},
pages = {282-291},
year = {2022},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2022.06.018},
url = {https://www.sciencedirect.com/science/article/pii/S0893608022002337},
author = {Gallil Maimon and Lior Rokach},
keywords = {Adversarial learning, NLP, Text classification, Universal adversarial attacks, Reinforcement learning},
abstract = {Discovering the existence of universal adversarial perturbations had large theoretical and practical impacts on the field of adversarial learning. In the text domain, most universal studies focused on adversarial prefixes which are added to all texts. However, unlike the vision domain, adding the same perturbation to different inputs results in noticeably unnatural inputs. Therefore, we introduce a new universal adversarial setup – a universal adversarial policy, which has many advantages of other universal attacks but also results in valid texts – thus making it relevant in practice. We achieve this by learning a single search policy over a predefined set of semantics preserving text alterations, on many texts. This formulation is universal in that the policy is successful in finding adversarial examples on new texts efficiently. Our approach uses text perturbations which were extensively shown to produce natural attacks in the non-universal setup (specific synonym replacements). We suggest a strong baseline approach for this formulation which uses reinforcement learning. Its ability to generalise (from as few as 500 training texts) shows that universal adversarial patterns exist in the text domain as well.}
}
```
