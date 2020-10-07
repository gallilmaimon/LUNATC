# imports
from operator import itemgetter
import numpy as np
import os
import pickle
import re

# Universal Sentence Encoder
import tensorflow as tf
import tensorflow_hub as hub

# synonyms
import nltk
from gensim.models import KeyedVectors
from nltk.corpus import stopwords

# re-translation
from translate import Translator

# misspelling
import nlpaug.model.char as nmc
import nlpaug.model.word_dict as nmwd
import random

LIB_DIR = os.path.abspath(__file__).split('text_xai')[0]

# region Constants ###################
# the Universal Sentence Encoder's TF Hub module
if os.path.exists(LIB_DIR + "/resources/tf_hub_modules/USE"):
    embed = hub.Module(LIB_DIR + "/resources/tf_hub_modules/USE")
else:
    print("Warning! local version of Universal Sentence Encoder model was not found trying to download. This may cause"
          "problems in runtime!")
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
# synonym word embeddings
word_vectors = KeyedVectors.load_word2vec_format(LIB_DIR +
                            "/resources/word_vectors/counter-fitted-vectors_formatted.txt", binary=False)

# sentence similarity - placed globally to avoid leakage
similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
similarity_message_encodings = embed(similarity_input_placeholder)
init = tf.global_variables_initializer()

# synonym action cache
synonym_act_dict = {}

# contractions dict
contractions = {
    "ain't": ["am not", "are not"],
    "aren't": ["are not", "am not"],
    "can't": ["can not"],
    "can't've": ["can not have"],
    "'cause": ["because"],
    "could've": ["could have"],
    "couldn't": ["could not"],
    "couldn't've": ["could not have"],
    "didn't": ["did not"],
    "doesn't": ["does not"],
    "don't": ["do not"],
    "hadn't": ["had not"],
    "hadn't've": ["had not have"],
    "hasn't": ["has not"],
    "haven't": ["have not"],
    "he'd": ["he had", "he would"],
    "he'd've": ["he would have"],
    "he'll": ["he shall", "he will"],
    "he'll've": ["he shall have", "he will have"],
    "he's": ["he has", "he is"],
    "how'd": ["how did"],
    "how'd'y": ["how do you"],
    "how'll": ["how will"],
    "how's": ["how has", "how is"],
    "i'd": ["I had", "I would"],
    "i'd've": ["I would have"],
    "i'll": ["I shall", "I will"],
    "i'll've": ["I shall have", "I will have"],
    "i'm": ["I am"],
    "i've": ["I have"],
    "isn't": ["is not"],
    "it'd": ["it had", "it would"],
    "it'd've": ["it would have"],
    "it'll": ["it shall", "it will"],
    "it'll've": ["it shall have", "it will have"],
    "it's": ["it has", "it is"],
    "let's": ["let us"],
    "ma'am": ["madam"],
    "mayn't": ["may not"],
    "might've": ["might have"],
    "mightn't": ["might not"],
    "mightn't've": ["might not have"],
    "must've": ["must have"],
    "mustn't": ["must not"],
    "mustn't've": ["must not have"],
    "needn't": ["need not"],
    "needn't've": ["need not have"],
    "o'clock": ["of the clock"],
    "oughtn't": ["ought not"],
    "oughtn't've": ["ought not have"],
    "shan't": ["shall not"],
    "sha'n't": ["shall not"],
    "shan't've": ["shall not have"],
    "she'd": ["she had", "she would"],
    "she'd've": ["she would have"],
    "she'll": ["she shall", "she will"],
    "she'll've": ["she shall have", "she will have"],
    "she's": ["she has", "she is"],
    "should've": ["should have"],
    "shouldn't": ["should not"],
    "shouldn't've": ["should not have"],
    "so've": ["so have"],
    "so's": ["so as", "so is"],
    "that'd": ["that would", "that had"],
    "that'd've": ["that would have"],
    "that's": ["that has", "that is"],
    "there'd": ["there had", "there would"],
    "there'd've": ["there would have"],
    "there's": ["there has", "there is"],
    "they'd": ["they had", "they would"],
    "they'd've": ["they would have"],
    "they'll": ["they shall", "they will"],
    "they'll've": ["they shall have", "they will have"],
    "they're": ["they are"],
    "they've": ["they have"],
    "to've": ["to have"],
    "wasn't": ["was not"],
    "we'd": ["we had", "we would"],
    "we'd've": ["we would have"],
    "we'll": ["we will"],
    "we'll've": ["we will have"],
    "we're": ["we are"],
    "we've": ["we have"],
    "weren't": ["were not"],
    "what'll": ["what shall", "what will"],
    "what'll've": ["what shall have", "what will have"],
    "what're": ["what are"],
    "what's": ["what has", "what is"],
    "what've": ["what have"],
    "when's": ["when has", "when is"],
    "when've": ["when have"],
    "where'd": ["where did"],
    "where's": ["where has", "where is"],
    "where've": ["where have"],
    "who'll": ["who shall", "who will"],
    "who'll've": ["who shall have", "who will have"],
    "who's": ["who has", "who is"],
    "who've": ["who have"],
    "why's": ["why has", "why is"],
    "why've": ["why have"],
    "will've": ["will have"],
    "won't": ["will not"],
    "won't've": ["will not have"],
    "would've": ["would have"],
    "wouldn't": ["would not"],
    "wouldn't've": ["would not have"],
    "y'all": ["you all"],
    "y'all'd": ["you all would"],
    "y'all'd've": ["you all would have"],
    "y'all're": ["you all are"],
    "y'all've": ["you all have"],
    "you'd": ["you had", "you would"],
    "you'd've": ["you would have"],
    "you'll": ["you shall", "you will"],
    "you'll've": ["you shall have", "you will have"],
    "you're": ["you are"],
    "you've": ["you have"]
    }

# stop words
STOPWORDS = stopwords.words('english')

# misspelling
keyboard_model = nmc.Keyboard(special_char=False, numeric=False, upper_case=False, lang='en', model_path=None)
misspell_words_model = nmwd.Spelling(LIB_DIR + '/resources/spelling/spelling_en.txt')

# end region constants


# region utility functions
def possible_actions(text):
    words = text.split()
    content_words = [i for i, w in enumerate(words) if w not in STOPWORDS and w in word_vectors]
    return content_words


def get_similarity(messages, sess):
    """ calculates the similarity between the first message and the others using Universal Sentence encoder"""
    message_embeddings = sess.run(similarity_message_encodings, feed_dict={similarity_input_placeholder: messages})
    corr = np.inner(message_embeddings[0], message_embeddings[1:])
    return corr


def score_method(sent, word, replacemnt_options, sess):
    """returns the index of the word which is best replacement"""
    if len(replacemnt_options) == 1:
        return 0
    sent_option_list = [sent] + [sent.replace(word, rep_option) for rep_option in replacemnt_options]
    return np.argmax(get_similarity(sent_option_list, sess))


def get_same_POS_replacements(text, word_index, replacement_options):
    # Get the list of words from the entire text
    words = text.split()
    # get all replacement options
    sents = [words[:word_index] + [rep_opt] + words[word_index + 1:] for rep_opt in replacement_options]
    # Identify the parts of speech
    tagged = nltk.pos_tag(words)
    tag = tagged[word_index][1]
    tags = nltk.pos_tag_sents(sents)
    return [i for i, x in enumerate(tags) if x[word_index][1] == tag]


def get_sentence_of_word_index(text: str, word_index: int):
    """
    this function gets a text and a word index, it returns only the sentence of the word given (by index), and updates
    the word  index accordingly
    :param text:
    :param word_index:
    :return:new_text, new_word_index - after taking only the wanted sentence
    """
    # sent_list = text.split(' .')
    # sent_list = [s + ' .' for s in sent_list if s != '']
    sent_list = nltk.sent_tokenize(text)
    sent_lens = [len(s.split()) - 0.001 for s in sent_list]  # the minus is used for index exactly at the end of a sentence
    cu = np.cumsum(sent_lens)
    idx = np.searchsorted(cu, word_index)
    new_text = sent_list[idx]
    if idx != 0:
        new_word_index = int(word_index - cu[idx - 1])
    else:
        new_word_index = word_index

    return new_text, new_word_index


def replace_word(text: str, word_ind: int, word: str):
    """
    replace the word at word_ind of text with a given word
    :param text: the original text
    :param word_ind: the index of the word to be replaced (when splitting by space)
    :param word: replacement word
    :return:
    """
    words = text.split()
    words[word_ind] = word
    return ' '.join(words)


def replace_special_chars(s):
    map_dict = {'’':"'", '…':'...', "–":'-', '“':'"', '—':'-', "‘":"'", '”':'"', "&amp;":"&", "&gt;":'', '&lt;':'', '\n':' ', '<br /><br />':' '}
    for key in map_dict:
        s = s.replace(key, map_dict[key])
    return s


def space_punctuation(s):
    s = re.sub(r'([0-9a-zA-Z])(\')([a-zA-Z])', r'\1____\3', s)  # save midword apostrephes
    s = re.sub(r'([a-zA-Z])(\.)([a-zA-Z])', r'\1~~~~\3', s)  # save midword .
    # space around punctuation
    s = re.sub(r'([a-zA-Z])([.,!?#*()/\[\];:"“”\-\'])', r'\1 \2', s)
    s = re.sub(r'([.,!?#*()/\[\];:"“”\-\'])([a-zA-Z])', r'\1 \2', s)

    s = re.sub(r'____', "'", s)  # return apostrhephes
    s = re.sub(r'~~~~', ".", s)  # return .

    return s


def clean_text(text):
    text = replace_special_chars(text)
    text = space_punctuation(text)

    # special case to surround "="+ with spaces even if it is joined to text
    text = re.sub(r'(=)+', ' \\1 ', text)

    text = text.lower()

    # remove extra whitespaces
    text = ' '.join(text.split())
    return text

# endregion utility functions


# region actions
def replace_with_synonym(text, word_index, sess, topn=10, word_sim_thresh=0.6,
                         sentence_sim_thresh=0.6, debug=False):
    """
    This function replaces the word at a given word_index (when splitting by spaces), of the given text with a synonym.
    The synonym is chosen by a series of steps. The whole process works only on the sentence in which the word is, for
    improved results and computational efficiency.
    0) if the word is a stop word or out of vocabulary no change is made
    1) generate synonym options by similar words according to cosine distance of word vectors curated for the task -
    http://mi.eng.cam.ac.uk/~nm480/naaclhlt2016.pdf
    2) remove words which aren't classified as having the same Part of Speech as the original (in context)
    3) replace the word with each of the options and compute the new sentence's similarity to the original, using the
    Universal Sentence Encoder - https://arxiv.org/abs/1803.11175
    4) return the most similar sentence
    this is similar to approach suggested in the paper - " Is BERT Really Robust? A Strong Baseline
    for Natural Language Attack on Text Classification and Entailment" - https://arxiv.org/abs/1907.11932, however it
    doesn't directly aim to "confuse" the Language model and therefore produces higher quality synonyms
    :param text: the original text
    :param word_index: the index of the word to be replaced
    :param sess: an initialised tensorflow session for runtime efficiency
    :param topn: how many candidates to consider as synonyms
    :param word_sim_thresh: how similar does a candidate synonym need to be in order to be considered
    :param sentence_sim_thresh: how similar does the new sentence need to be to the original
    :param debug: a flag for extra printed information
    :return: the new text after the replacement
    """
    # work with single sentence only (that of the given word)
    new_text, new_word_index = get_sentence_of_word_index(text, word_index)
    # look in cache of previously calculated actions
    if (new_text, int(new_word_index)) in synonym_act_dict:
        rep_word = synonym_act_dict[(new_text, int(new_word_index))]
        return replace_word(text, word_index, rep_word)

    # Get the list of words from the entire text
    words = new_text.split()
    word = words[new_word_index]
    print('text', text) if debug else ''
    print('word', word) if debug else ''

    # if word not in vocabulary or in stopwords
    if word in STOPWORDS or word not in word_vectors:
        synonym_act_dict[(new_text, int(new_word_index))] = word
        return text

    # find synonym options
    rep_options = word_vectors.most_similar(positive=[word], topn=topn)
    print('all', rep_options) if debug else ''
    rep_options = [word for word, sim_score in rep_options if sim_score > word_sim_thresh]
    print('thresh', rep_options) if debug else ''

    # no good enough synonyms
    if len(rep_options) == 0:
        synonym_act_dict[(new_text, int(new_word_index))] = word
        return text

    # get only those with same POS
    same_pos_inds = get_same_POS_replacements(new_text, new_word_index, rep_options)
    print('same pos inds', same_pos_inds) if debug else ''
    if len(same_pos_inds) == 0:
        synonym_act_dict[(new_text, int(new_word_index))] = word
        return text
    rep_options = itemgetter(*same_pos_inds)(rep_options)
    if type(rep_options) == str:
        rep_options = list([rep_options])
    else:
        rep_options = list(rep_options)
    print('same POS', rep_options) if debug else ''

    # get sentence similarity to original
    sent_options = []
    for opt in rep_options:
        words[new_word_index] = opt
        sent_options.append(' '.join(words))
    sentence_similarity = get_similarity([new_text] + sent_options, sess)
    print('sentence similarity', sentence_similarity) if debug else ''
    best_option = np.argmax(sentence_similarity)
    print(best_option) if debug else ''

    if sentence_similarity[best_option] >= sentence_sim_thresh:
        synonym_act_dict[(new_text, int(new_word_index))] = rep_options[best_option]
        return replace_word(text, word_index, rep_options[best_option])

    synonym_act_dict[(text, int(word_index))] = word
    return text


def replace_with_synonym_all_text(text, word_index, sess, topn=10, word_sim_thresh=0.6,
                                  sentence_sim_thresh=0.6, debug=False):
    """
    This function replaces the word at a given word_index (when splitting by spaces), of the given text with a synonym.
    The synonym is chosen by a series of steps. The whole process works on the whole text, as opposed to
    replace_with_synonym
    0) if the word is a stop word or out of vocabulary no change is made
    1) generate synonym options by similar words according to cosine distance of word vectors curated for the task -
    http://mi.eng.cam.ac.uk/~nm480/naaclhlt2016.pdf
    2) remove words which aren't classified as having the same Part of Speech as the original (in context)
    3) replace the word with each of the options and compute the new sentence's similarity to the original, using the
    Universal Sentence Encoder - https://arxiv.org/abs/1803.11175
    4) return the most similar sentence
    this is similar to approach suggested in the paper - " Is BERT Really Robust? A Strong Baseline
    for Natural Language Attack on Text Classification and Entailment" - https://arxiv.org/abs/1907.11932, however it
    doesn't directly aim to "confuse" the Language model and therefore produces higher quality synonyms
    :param text: the original text
    :param word_index: the index of the word to be replaced
    :param sess: an initialised tensorflow session for runtime efficiency
    :param topn: how many candidates to consider as synonyms
    :param word_sim_thresh: how similar does a candidate synonym need to be in order to be considered
    :param sentence_sim_thresh: how similar does the new sentence need to be to the original
    :param debug: a flag for extra printed information
    :return: the new text after the replacement
    """
    # work with single sentence only (that of the given word)
    print('----------------------------')
    print('before, ', text, word_index)
    text, word_index, sents, sent_ind = get_sentence_of_word_index(text, word_index)
    print('after, ', text, word_index)
    # look in cache of previously calculated actions
    if (text, int(word_index)) in synonym_act_dict:
        # print(f'len of syn dict {len(synonym_act_dict)}')
        return synonym_act_dict[(text, int(word_index))]

    # Get the list of words from the entire text
    words = text.split()
    word = words[word_index]
    print('text', text) if debug else ''
    print('word', word) if debug else ''

    # if word not in vocabulary or in stopwords
    if word in STOPWORDS or word not in word_vectors:
        synonym_act_dict[(text, int(word_index))] = text
        return text

    # find synonym options
    rep_options = word_vectors.most_similar(positive=[word], topn=topn)
    print('all', rep_options) if debug else ''
    rep_options = [word for word, sim_score in rep_options if sim_score > word_sim_thresh]
    print('thresh', rep_options) if debug else ''

    # no good enough synonyms
    if len(rep_options) == 0:
        synonym_act_dict[(text, int(word_index))] = text
        return text

    # get only those with same POS
    same_pos_inds = get_same_POS_replacements(text, word_index, rep_options)
    if len(same_pos_inds) == 0:
        synonym_act_dict[(text, int(word_index))] = text
        return text
    rep_options = itemgetter(*same_pos_inds)(rep_options)
    if type(rep_options) == str:
        rep_options = list([rep_options])
    else:
        rep_options = list(rep_options)
    print('same POS', rep_options) if debug else ''

    # get sentence similarity to original
    sent_options = []
    for opt in rep_options:
        words[word_index] = opt
        sent_options.append(' '.join(words))
    sentence_similarity = get_similarity([text] + sent_options, sess)
    print('sentence similarity', sentence_similarity) if debug else ''
    best_option = np.argmax(sentence_similarity)
    print(best_option) if debug else ''

    if sentence_similarity[best_option] >= sentence_sim_thresh:
        synonym_act_dict[(text, int(word_index))] = sent_options[best_option]
        return sent_options[best_option]
    synonym_act_dict[(text, int(word_index))] = text
    return text


def replace_with_synonym_greedy(text, word_index, text_model, sess, topn=10, word_sim_thresh=0.6,
                                sentence_sim_thresh=0.6, debug=False):
    """
    This function replaces the word at a given word_index (when splitting by spaces), of the given text with a synonym.
    The synonym is chosen by a series of steps as presented in the paper - " Is BERT Really Robust? A Strong Baseline
    for Natural Language Attack on Text Classification and Entailment" - https://arxiv.org/abs/1907.11932
    :param text: the original text
    :param word_index: the index of the word to be replaced
    :param text_model: the language model being "attacked"
    :param sess: an initialised tensorflow session for runtime efficiency
    :param topn: how many candidates to consider as synonyms
    :param word_sim_thresh: how similar does a candidate synonym need to be in order to be considered
    :param sentence_sim_thresh: how similar does the new sentence need to be to the original
    :param debug: a flag for extra printed information
    :return: the new text after the replacement
    """
    # look in cache of previously calculated actions
    # if (text, int(word_index)) in synonym_act_dict:
    #     # print(f'len of syn dict {len(synonym_act_dict)}')
    #     return synonym_act_dict[(text, int(word_index))]

    # Get the list of words from the entire text
    words = text.split()
    word = words[word_index]
    print('text', text) if debug else ''
    print('word', word) if debug else ''

    # if word not in vocabulary
    if word not in word_vectors:
        synonym_act_dict[(text, int(word_index))] = text
        return text

    # find synonym options
    rep_options = word_vectors.most_similar(positive=[word], topn=topn)
    print('all', rep_options) if debug else ''
    rep_options = [word for word, sim_score in rep_options if sim_score > word_sim_thresh]
    print('thresh', rep_options) if debug else ''

    # no good enough synonyms
    if len(rep_options) == 0:
        synonym_act_dict[(text, int(word_index))] = text
        return text

    # get only those with same POS
    same_pos_inds = get_same_POS_replacements(text, word_index, rep_options)
    if len(same_pos_inds) == 0:
        synonym_act_dict[(text, int(word_index))] = text
        return text
    rep_options = itemgetter(*same_pos_inds)(rep_options)
    if type(rep_options) == str:
        rep_options = list([rep_options])
    else:
        rep_options = list(rep_options)
    print('same POS', rep_options) if debug else ''

    # get sentence similarity to original
    sent_options = []
    for opt in rep_options:
        words[word_index] = opt
        sent_options.append(' '.join(words))
    sentence_similarity = get_similarity([text] + sent_options, sess)
    print('sentence similarity', sentence_similarity) if debug else ''
    cand_mask = (sentence_similarity >= sentence_sim_thresh)
    print('cand mask', cand_mask) if debug else ''
    if cand_mask.sum() == 1:
        synonym_act_dict[(text, int(word_index))] = [i for (i, v) in zip(sent_options, cand_mask) if v][0]
        return [i for (i, v) in zip(sent_options, cand_mask) if v][0]
    elif cand_mask.sum() > 1:
        sent_options = [i for (i, v) in zip(sent_options, cand_mask) if v]
        print('sent options: ', sent_options) if debug else ''
        sentence_similarity = sentence_similarity[cand_mask]
        orig_probs = text_model.predict_proba(text)[0]
        orig_pred = np.argmax(orig_probs)
        new_probs = [text_model.predict_proba(new_sent)[0][orig_pred] for new_sent in sent_options]
        print('new probs: ', new_probs) if debug else ''
        changed_class = list(map(lambda x: x < 0.5, new_probs))
        if sum(changed_class) >= 1:
            cur_sent_options = [i for (i, v) in zip(sent_options, changed_class) if v]
            synonym_act_dict[(text, int(word_index))] = cur_sent_options[np.argmax(sentence_similarity[changed_class])]
            print('result: ', cur_sent_options[np.argmax(sentence_similarity[changed_class])]) if debug else ''
            return cur_sent_options[np.argmax(sentence_similarity[changed_class])]
        synonym_act_dict[(text, int(word_index))] = sent_options[np.argmin(new_probs)]
        print('result2: ', sent_options[np.argmin(new_probs)]) if debug else ''
        return sent_options[np.argmin(new_probs)]

    synonym_act_dict[(text, int(word_index))] = text
    return text


def remove_word(text, word_index):
    words = text.split()
    return " ".join(words[:word_index] + words[word_index+1:])


def remove_contractions(text, sess):
    for word in text.split():
        word_lower = word.lower()
        if word_lower in contractions:
            options = contractions[word_lower]
            ind = score_method(text, word, options, sess)
            text = text.replace(word, options[ind])
    return text


def retranslate(text, a, locks_dict):
    lang_dic = {0: 'hi',
                1: 'fr',
                2: 'zh-Hans',
                3: 'he'}
    # translate to non-english
    locks_dict[f'en2{lang_dic[a]}'].acquire()
    with open(f'{LIB_DIR}/resources/retranslation_files/en2{lang_dic[a]}.pkl', 'rb') as f:
        en2lang_dict = pickle.load(f)
    if text in en2lang_dict:
        translation = en2lang_dict[text]
    else:
        en2lang = Translator(from_lang='en', to_lang=lang_dic[a], provider='microsoft', secret_access_key='ACCESS_KEY')
        translation = en2lang.translate(text)
        en2lang_dict[text] = translation
        with open(f'{LIB_DIR}/resources/retranslation_files/en2{lang_dic[a]}.pkl', 'wb') as f:
            pickle.dump(en2lang_dict, f)
    locks_dict[f'en2{lang_dic[a]}'].release()

    # translate back to english
    locks_dict[f'{lang_dic[a]}2en'].acquire()
    with open(f'{LIB_DIR}/resources/retranslation_files/{lang_dic[a]}2en.pkl', 'rb') as f:
        lang2en_dict = pickle.load(f)
    if translation in lang2en_dict:
        retrans = lang2en_dict[translation]
    else:
        lang2en = Translator(from_lang=lang_dic[a], to_lang="en", provider='microsoft', secret_access_key='ACCESS_KEY')
        retrans = lang2en.translate(translation)
        lang2en_dict[translation] = retrans
        with open(f'{LIB_DIR}/resources/retranslation_files/{lang_dic[a]}2en.pkl', 'wb') as f:
            pickle.dump(lang2en_dict, f)
    locks_dict[f'{lang_dic[a]}2en'].release()

    return clean_text(retrans)


def qwerty_misspell(word):
    allowed_chars = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u',
                     'v', 'w', 'x', 'y', 'z'}
    chars = list(word)

    # only replace letter characters
    alpha_chars_ind = [i for i in range(len(chars)) if chars[i] in allowed_chars]
    if len(alpha_chars_ind) == 0:
        return word

    char_ind = alpha_chars_ind[int(0.5*len(alpha_chars_ind))]  # change the middle charcter
    chars[char_ind] = keyboard_model.predict(chars[char_ind])[0]  # select the first replacement
    return ''.join(chars)


def misspell(text, word_ind):
    words = text.split()
    word = words[word_ind]

    # first try dictionary misspelling

    possible_words = misspell_words_model.predict(word)
    if possible_words is not None:
        possible_words = [w for w in possible_words if "'" not in w]
        if len(possible_words) > 0:
            new_word = possible_words[0].lower()  # select the first replacement
            words[word_ind] = new_word
            return ' '.join(words)

    # if doesn't appear in predefined misspelling
    new_word = qwerty_misspell(word)
    words[word_ind] = new_word
    return ' '.join(words)
# endregion actions
