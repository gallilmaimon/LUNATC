# A Universal Adversarial Policy for Text Classifiers

This repository is the official implementation of LUNATC from "A Universal Adversarial Policy for Text Classifiers". This version will be further documented for official release.

## Requirements

To download needed resources:

1) downloading the word vectors for synonyms
* download the counter-fitted-vectors.txt.zip from the following [git](https://github.com/nmrksic/counter-fitting/tree/master/word_vectors)
* unzip the file into the resources/word_vectors folder
* run the [notebook](https://github.com/gallilmaimon/LUNATC/blob/master/resources/word_vectors/format_wordvectors.ipynb) to format the weights in the needed format. 

2) install the tf_hub module for the Universal Sentence Encoder
```setup
# make directory
mkdir resources/tf_hub_modules
mkdir resources/tf_hub_modules/USE
# download model
wget "https://tfhub.dev/google/universal-sentence-encoder/2?tf-hub-format=compressed" -O "USE.tar.gz"
tar -C tf_hub_modules/USE -xzvf USE.tar.gz
# cleanup
rm USE.tar.gz
```

3) install requirements:

```setup
pip install -r requirements.txt
```

## Setting up data & training the attacked model

* if using the pre-trained models and the sample, this stage can be skipped. 

Downloading the datasets:

1) IMDB - download from this [link](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz), and extract the zip in the data/aclImdb folder.

2) Toxic-Wikipedia - download test_labels.csv.zip, test.csv.zip, train.csv.zip from [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data), and extract them in the data/toxic folder.

Preprocessing the texts, and training the model:
- The [data/preprocess-data](https://github.com/gallilmaimon/LUNATC/blob/master/data/preprocess%20data.ipynb) shows how to get from each of the datasets downloaded to a single format suitable for training a BERT or word-LSTM classifier.

- We then use the following [notebook](https://github.com/gallilmaimon/LUNATC/blob/master/train%20BERT.ipynb) for training a BERT classifier on these datasets, or this [notebook](https://github.com/gallilmaimon/LUNATC/blob/master/train%20word%20LSTM.ipynb) for training word-LSTM.

- Now use [data/preprocess-data](https://github.com/gallilmaimon/LUNATC/blob/master/data/preprocess%20data.ipynb) for generating the same data samples used in the paper.

## generating the agent's train and test sets
Defintion of the agent training and test sets. To generate them from scratch (and see how the texts were chosen) one needs to first attack all model test texts using our imolementation of [textfooler](https://github.com/gallilmaimon/LUNATC/blob/master/src/Attacks/textfooler_attack.py), [PWWS](https://github.com/gallilmaimon/LUNATC/blob/master/src/Attacks/textfooler_attack.py) (change the attack type parameter), and [randomly attack](https://github.com/gallilmaimon/LUNATC/blob/master/src/Attacks/DQN_attack.py) for 100 rounds (tweak accordingly). Then take only the texts which textfooler managed to attack, and take the first *train_size* of them as training samples, and all other succesful attacks are used as test, while sometimes selecting only a subset for runtime considerations.

The pre-calculated indices of each set appear in this [file](), for your ease of use.

## Performing the attacks

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
