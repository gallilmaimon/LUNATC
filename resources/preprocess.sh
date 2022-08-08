#!/bin/sh

## region synonym vectors
# downloading word vectors
echo "Downloading counter fitted word vectors for synonym"
cd resources/word_vectors || return
wget -O counter-fitted-vectors.txt.zip https://github.com/nmrksic/counter-fitting/blob/master/word_vectors/counter-fitted-vectors.txt.zip?raw=true
unzip counter-fitted-vectors.txt.zip  # this requires the unzip utility to be installed, download with apt-get or unzip manually if needed

# preprocess vectors
python format_wv.py

# cleanup
rm counter-fitted-vectors.txt.zip
rm counter-fitted-vectors.txt
## endregion

## region USE module
echo "Downloading Universal Sentence Encoder tf hub module"
# make directory
cd ..
mkdir tf_hub_modules
mkdir tf_hub_modules/USE
# download tf_hub model
wget "https://tfhub.dev/google/universal-sentence-encoder/3?tf-hub-format=compressed" -O "USE.tar.gz"
tar -C tf_hub_modules/USE -xzvf USE.tar.gz
# cleanup
rm USE.tar.gz
## endregion

## region Glove vectors
echo "Downloading Glove word vectors"
cd word_vectors || return
wget wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip  # this requires the unzip utility to be installed, download with apt-get or unzip manually if needed
rm glove.6B.zip
rm glove.6B.50d.txt
rm glove.6B.100d.txt
rm glove.6B.300d.txt
## endregion

# region Spacy English
echo "download Spacy en_core_web_sm"
python -m spacy download en_core_web_sm
## endregion