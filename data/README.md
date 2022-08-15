### Manually parsing data & training attacked models
If you wish to change the parsing if the dataset, or change the model training scheme you can do so using the following methods.

#### Downloading the (original, unparsed) datasets:
1) IMDB - download from this [link](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz), and extract the zip in the data/aclImdb folder.

2) Toxic-Wikipedia - download test_labels.csv.zip, test.csv.zip, train.csv.zip from [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data), and extract them in the data/toxic folder.

3) PUBMED - for full reproducibility we suggest you use our provided parsed dataset. If you want te reproduce it (with updated files for instance) download the new xmls from pubmed and use the code in data/parse Pumbed.ipynb.

#### Preprocessing the texts, and training the model:
- The [data/preprocess-data](https://github.com/gallilmaimon/LUNATC/blob/master/data/preprocess%20data.ipynb) shows how to get from each of the datasets downloaded to a single format suitable for training a BERT or word-LSTM classifier.

- We then use the following [notebook](https://github.com/gallilmaimon/LUNATC/blob/master/train%20BERT.ipynb) for training a BERT classifier on these datasets, or this [notebook](https://github.com/gallilmaimon/LUNATC/blob/master/train%20word%20LSTM.ipynb) for training word-LSTM.

- Now use [data/preprocess-data](https://github.com/gallilmaimon/LUNATC/blob/master/data/preprocess%20data.ipynb) for generating the same data samples used in the paper.

#### Generating the agent's train and test sets
Defintion of the agent training and test sets. To generate them from scratch (and see how the texts were chosen) one needs to first attack all model test texts using our imolementation of [textfooler](https://github.com/gallilmaimon/LUNATC/blob/master/src/Attacks/textfooler_attack.py), [PWWS](https://github.com/gallilmaimon/LUNATC/blob/master/src/Attacks/textfooler_attack.py) (change the attack type parameter), and [randomly attack](https://github.com/gallilmaimon/LUNATC/blob/master/src/Attacks/DQN_attack.py) for 100 rounds (tweak accordingly). Then take only the texts which textfooler managed to attack, and take the first *train_size* of them as training samples, and all other succesful attacks are used as test, while sometimes selecting only a subset for runtime considerations.

The pre-calculated indices of each set appear in this attack-indices file, for your ease of use.
