# Data pre-processing & training attacked classifiers
This README file describes the different ways to download all the datasets described in the paper, it describes how to pre-process them (or other new datasets) into the proper format. It also explains how to train specific classifiers on the datasets in order to attack (BERT, XLNet, word-LSTM), however based on those examples other models can be trained.

### Downloading the (original, unparsed) datasets:
1) IMDB - download from this [link](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz), and extract the zip in the data/aclImdb folder.

2) Toxic-Wikipedia - download test_labels.csv.zip, test.csv.zip, train.csv.zip from [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data), and extract them in the data/toxic folder.

3) PUBMED - this is a newly introduced dataset, available for download at the following [link](bla). This can also be parsed from scratch based on the raw pubmed XMLs, however parsing the entire dataset from the raw files is compute intensive, and requires large storage. If you wish to reproduce it, or update it - download the new xmls from pubmed.

4) MNLI - download from this official [link](https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip), and extract the zip into data/mnli folder.

### Preprocessing the texts, and training the model:
- The [data/preprocess-data](https://github.com/gallilmaimon/LUNATC/blob/master/data/preprocess%20data.ipynb) shows how to get from each of the datasets downloaded to a single format suitable for training a BERT or word-LSTM classifier. This is made of 2 csv files (train and test) with 2 columns: 'content' - for the text, and 'label' for an int value for the true class.

- We then train a classifier on these datasets, using the following commands (based on which model to train and which dataset):
```
# PUBMED + BERT
python data/train_transformer.py --data_path data/pubmed/pubmed
# PUBMED + LSTM 
python data/train_LSTM.py --data_path data/pubmed/pubmed --n_epochs 2 --lr 3e-4
# Toxic + LSTM
python data/train_LSTM.py --data_path data/toxic/toxic  --n_epochs 4 --lr 3e-4
# Toxic + BERT
python data/train_transformer.py  --data_path data/toxic/toxic
# IMDB + LSTM
python data/train_LSTM.py --data_path data/aclImdb/imdb --lr 3e-4
# IMDB + BERT
python data/train_transformer.py  --data_path data/aclImdb/imdb
# IMDB + XLNet
python data/train_transformer.py  --data_path data/aclImdb/imdb --model xlnet
# MNLI + BERT
python data/train_transformer.py  --data_path data/mnli/mnli --n_epochs 4 --n_classes 3
```

- We infer on the test set to get the sample used for attacking, using the following commands (based on which model to train and which dataset):
```
# PUBMED + BERT
python data/train_transformer.py --data_path data/pubmed/pubmed --mode infer
# PUBMED + LSTM
python data/train_LSTM.py --data_path data/pubmed/pubmed  --mode infer
# Toxic + LSTM
python data/train_LSTM.py --data_path data/toxic/toxic --mode infer
# Toxic + BERT
python data/train_transformer.py  --data_path data/toxic/toxic --mode infer
# IMDB + LSTM
python data/train_LSTM.py --data_path data/aclImdb/imdb --mode infer
# IMDB + BERT
python data/train_transformer.py  --data_path data/aclImdb/imdb --mode infer
# IMDB + XLNet
python data/train_transformer.py  --data_path data/aclImdb/imdb --model xlnet --mode infer
# MNLI + BERT
python data/train_transformer.py  --data_path data/mnli/mnli --n_classes 3  --mode infer
```

### Generating the agent's train and test sets
- Now use [data/preprocess-data](https://github.com/gallilmaimon/LUNATC/blob/master/data/preprocess%20data.ipynb) for generating the same data samples used in the paper. This essentially filters texts by length and orders them by classes and predictions.

- If attacking, with baselines you can attack the entire set (the output csv from the previous stage) as explained in the main readme in the baselines section. However, for LUNATC, we also calculate specific indices in which a succesful attack was found to exist (by one of the baselines) and only use those indices for training and evaluating the universal adversarial policy. This is explained in [data/preprocess-data](https://github.com/gallilmaimon/LUNATC/blob/master/data/preprocess%20data.ipynb).
