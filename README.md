# A Universal Adversarial Policy for Text Classifiers

This repository is the official implementation of LUNATC from "A Universal Adversarial Policy for Text Classifiers". https://doi.org/10.1016/j.neunet.2022.06.018 

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
| IMDB (aclImdb)          | 93.98, [model](https://drive.google.com/file/d/1MtEzBmLmSn4ad-EefalzBOYyZZzltq71/view?usp=sharing), data                  |  85.70, model, data                 |
| Toxic-Wikipedia (toxic) | 91.80, model, data                  |  92.77, model, data                 |
| Pubmed (pubmed)         | 96.75, model, data                  |  95.71, model, data                 |
| MNLI (mnli)             | 83.72, model, data                  | -                                   |


* For training from scratch, switching classifiers, using new datasets, changing data preprocess or anything else like that, see the following readme [data/README.md](https://github.com/gallilmaimon/LUNATC/blob/master/data/README.md).

## Performing the attacks
### Baselines

### LUNATC
To attack using LUNATC described in the paper, change the [configuration file](https://github.com/gallilmaimon/LUNATC/blob/master/src/Config/DQN_constants.yml) according to the following table, and the dataset (indicated by the base path), and the indices (based on the previous section). Seeds used are 42, 43, 44. Model type depends on the attacked model. Other fields should remain the same as original.

| Parameter\Train Size | 500       | 750   | 950   | 25k   | 50k   |
| :--------------------|:----------| :-----| :-----| :-----| :-----|
| Num Episodes         | 25000     | 25000 | 25000 | 50000 | 50000 |
| Memory Size          | 10000     | 15000 | 20000 | 25000 | 25000 |
| Target Update        | 2500      | 3750  | 5000  | 12500 | 12500 |
| Eps Decay            | 7500      | 11250 | 15000 | 37500 | 37500 |
| Num Rounds (in code) | 2         |    4  |  4    | 3     | 1     |


Afterwards run this command:
```LUNATC universal attack
python LUNATC/src/Attacks/DQN_attack.py 
```

To attack using Genfooler variants use the following [script](https://github.com/gallilmaimon/LUNATC/blob/master/src/Attacks/genfooler_attack.py).
The results for Random, PWWS, Textfooler were already acheived in the previous section.

## Evaluation

To evaluate the different approaches, use the [analyse experiments notebook](https://github.com/gallilmaimon/LUNATC/blob/master/analyse%20experiments.ipynb).
- For analysing the individual attacks seperately look at the "single sentence evaluation" section
- For analysing statistics of the individual attacks look at "multiple sentence evaluation" section
- For analysing the universal policy look at the "train test evaluation" section, it shows how to calculate all the graphs showed in the paper.


## Citation
If you found this work useful, please cite the following related article:

```@article{MAIMON2022282,
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
