# LUNATC: Learning a Universal Adversarial Policy for Text Classifiers

This repository is the official implementation of LUNATC: Learning a Universal Adversarial Policy for Text Classifiers. This version will be further documented for official release.

## Requirements

To download needed resources:

1) downloading the word vectors for synonyms
* download the counter-fitted-vectors.txt.zip from the following [git](https://github.com/nmrksic/counter-fitting/tree/master/word_vectors)
* unzip the file into the resources/word_vectors folder
* run the [notebook](https://github.com/gallilmaimon/text-xai/blob/master/resources/word_vectors/format_wordvectors.ipynb) to format the weights in the needed format. 

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
- The [data/preprocess-data](https://github.com/gallilmaimon/text-xai/blob/master/data/preprocess%20data.ipynb) shows how to get from each of the datasets downloaded to a single format suitable for training a BERT language model.

- We then use the following [notebook](https://github.com/gallilmaimon/text-xai/blob/master/train%20e2e%20BERT.ipynb) for training a BERT classifier on these datasets

- Now use [data/preprocess-data](https://github.com/gallilmaimon/text-xai/blob/master/data/preprocess%20data.ipynb) for generating the same data samples used in the paper.

## Spliting the dataset into groups
The splitting into sets (easy, converged etc.), with the performed ReLAX individual attacks are:
### Toxic-TP
easy: 2, 17, 35, 40, 45, 47, 52, 70, 71, 73, 79, 82, 84, 91, 97, 100, 115, 119, 121, 122, 124  
converged: 1, 4, 5, 13, 51, 58, 61, 64, 78, 87, 93, 109  
successful: 0, 12, 18, 20, 21, 24, 31, 56, 74, 76, 88, 95, 108, 113, 114, 120  
unsuccessful: 3, 6, 7, 8, 9, 10, 11, 14, 15, 16, 19, 22, 23, 25, 26, 27, 28, 29, 30, 32, 33, 34, 36, 37, 38, 39, 41, 42, 43, 44, 46, 48, 49, 50, 53, 54, 55, 57, 59, 60, 62, 63, 65, 66, 67, 68, 69, 72, 75, 77, 80, 81, 83, 85, 86, 89, 90, 92, 94, 96, 98, 99, 101, 102, 103, 104, 105, 106, 107, 110, 111, 112, 116, 117, 118,  123  

### IMDB - TP
easy: 2, 12, 19, 24, 41, 67, 78, 83, 88, 92, 102  
converged: 4, 6, 13, 27, 28, 34, 35, 56, 60, 68, 77, 93, 96, 104, 119  
successful: 0, 29, 32, 33, 36, 45, 49, 52, 53, 59, 62, 64, 65, 66, 72, 79, 80, 85, 87, 89, 95, 107, 108, 109, 112, 113, 115, 121, 124  
unsuccessful: all the rest up to 124 inclusive  

### IMDB - TN
easy: 309, 313, 352, 392, 430, 431, 436  
converged: 289, 290, 302, 312, 323, 329, 330, 336, 346, 351, 356, 360, 361, 363, 364, 365 , 370, 371, 374, 386, 387, 414, 422, 429  
successful: 280, 284, 292, 307, 327, 333, 334, 341, 342, 347, 348, 349, 367, 373, 388, 390, 391, 394, 396, 397, 402, 403, 419, 421, 427, 433, 434, 435  
unsuccessful:  all the rest between 278 and 437 inclusive  

If calculating the sets (easy, converged etc.) from scratch one needs to first attack using ReLAX individual attack (based on the next part) on all indices in the set. Then use the [notebook](https://github.com/gallilmaimon/text-xai/blob/sync_experiments/analyse%20experiments.ipynb) in the 'individual sentence evaluation' part, under - "calculate which examples are - easy, converged, successful and un-successful"


## Preforming the attacks

To attack using LUNATC described in the paper, change the [configuration file](https://github.com/gallilmaimon/text-xai/blob/master/text_xai/Config/constants.yml) to the contents of [universal_lunatc_constants.yml](https://github.com/gallilmaimon/text-xai/blob/sync_experiments/text_xai/Config/universal_lunatc_constants.yml), and change the dataset (according to the base path), and the indices (based on the previous section). Other fields should remain the same. 
Afterwards run this command:

```LUNATC universal attack
python text-xai/Agents/ReLAX_attack.py 
```

To attack using individual attacks with ReLAX, change the [configuration file](https://github.com/gallilmaimon/text-xai/blob/master/text_xai/Config/constants.yml) to the contents of [individual_relax_constants.yml](https://github.com/gallilmaimon/text-xai/blob/sync_experiments/text_xai/Config/individual_relax_constants.yml), and change the dataset (according to the base path), and the indices (based on the previous section). Other fields should remain the same. Then run this command:

```Relax individual attack
python text-xai/Agents/ReLAX_attack.py 
```


## Evaluation

To evaluate the different approaches, use the [analyse experiments notebook](https://github.com/gallilmaimon/text-xai/blob/master/analyse%20experiments.ipynb).
- For analysing the individual attacks seperately look at the "single sentence evaluation" section
- For analysing statistics of the individual attacks look at "multiple sentence evaluation" section
- For analysing the universal policy look at the "train test evaluation" section, it shows how to calculate all the graphs showed in the paper.
