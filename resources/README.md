# Resources

This document describes the resources provided here (or downloaded and parsed here) in use in the repository. 
This includes pretrained word vectors, and models of different kinds, as well as common english spelling mistakes used for the mispelling "augmentation".

## Counter fitted vectors
These word vectors are based on:

And have been shown to be good for generating synonym candidates.

## Glove Vectors
The Glove vectors are word vectors which are used to represent semantic meaning in various context, we use those with 6 billion tokens. and a vector size of 200.
For more info see https://nlp.stanford.edu/projects/glove/

## Universal Sentence Encoder
We download the pretrained tensorflow module of Universal Sentence Encoder 3 locally, for easier, quicker loading in runtime.
For more info see https://tfhub.dev/google/universal-sentence-encoder/4

## Spacy en_core_web_sm
This is needed to use Spacy's abilities in English.


## Spelling
The file [spelling/spelling_en.txt](https://github.com/gallilmaimon/LUNATC/blob/master/resources/spelling/spelling_en.txt) has a common list of English misppelings of words curated from the internet, this is used for "augmentation".