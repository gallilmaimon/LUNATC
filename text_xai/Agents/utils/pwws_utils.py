import numpy as np
import pandas as pd
import copy
from functools import partial
from collections import defaultdict

# text modelling
from nltk.corpus import wordnet as wn
import spacy

nlp = spacy.load('en_core_web_sm')

NE_type_dict = {
    'PERSON': defaultdict(int),  # People, including fictional.
    'NORP': defaultdict(int),  # Nationalities or religious or political groups.
    'FAC': defaultdict(int),  # Buildings, airports, highways, bridges, etc.
    'ORG': defaultdict(int),  # Companies, agencies, institutions, etc.
    'GPE': defaultdict(int),  # Countries, cities, states.
    'LOC': defaultdict(int),  # Non-GPE locations, mountain ranges, bodies of water.
    'PRODUCT': defaultdict(int),  # Object, vehicles, foods, etc.(Not services)
    'EVENT': defaultdict(int),  # Named hurricanes, battles, wars, sports events, etc.
    'WORK_OF_ART': defaultdict(int),  # Titles of books, songs, etc.
    'LAW': defaultdict(int),  # Named documents made into laws.
    'LANGUAGE': defaultdict(int),  # Any named language.
    'DATE': defaultdict(int),  # Absolute or relative dates or periods.
    'TIME': defaultdict(int),  # Times smaller than a day.
    'PERCENT': defaultdict(int),  # Percentage, including "%".
    'MONEY': defaultdict(int),  # Monetary values, including unit.
    'QUANTITY': defaultdict(int),  # Measurements, as of weight or distance.
    'ORDINAL': defaultdict(int),  # "first", "second", etc.
    'CARDINAL': defaultdict(int),  # Numerals that do not fall under another type.
}


def softmax(x, axis=0):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis)


def get_wordnet_pos(token):
    pos = token.tag_[0].lower()
    if pos == 'j':
        return 'a'
    elif pos in ['r', 'n', 'v']:
        return pos


def synonym_filter_fn(token, synonym):
    if (len(synonym.text.split()) > 2 or (  # the synonym produced is a phrase
            synonym.lemma == token.lemma) or (  # token and synonym are the same
            synonym.tag != token.tag) or (  # the pos of the token synonyms are different
            token.text.lower() == 'be')):  # token is be
        return False
    return True


def get_synonym_options(token):
    wordnet_pos = get_wordnet_pos(token)
    wordnet_synonyms = []

    synsets = wn.synsets(token.text, pos=wordnet_pos)
    for synset in synsets:
        wordnet_synonyms.extend(synset.lemmas())

    synonyms = []
    for wordnet_synonym in wordnet_synonyms:
        spacy_synonym = nlp(wordnet_synonym.name().replace('_', ' '))[0]
        synonyms.append(spacy_synonym)

    synonyms = filter(partial(synonym_filter_fn, token), synonyms)
    return set(synonyms)


def calculate_word_saliency(text_model, words):
    orig_probs = softmax(text_model.predict_proba(' '.join(words))[0])
    orig_pred = np.argmax(orig_probs)
    orig_prob = orig_probs[orig_pred]
    new_sents = [' '.join(words[:i] + ['<oov>'] + words[i + 1:]) for i in range(len(words))]
    return softmax(np.array([orig_prob - softmax(text_model.predict_proba(new_sent)[0])[orig_pred]
                             for new_sent in new_sents]))


def delta_p_star(text_model, words, i, rep_options):
    rep_options = [word if type(word) == str else word.text for word in rep_options]
    orig_probs = softmax(text_model.predict_proba(' '.join(words))[0])
    orig_pred = np.argmax(orig_probs)
    orig_prob = orig_probs[orig_pred]
    new_texts = []
    for word in rep_options:
        words[i] = word
        new_texts.append(' '.join(words))

    new_probs = np.concatenate([softmax(text_model.predict_proba(new_text), axis=1)
                                for new_text in new_texts])[:, orig_pred]
    chosen_ind = np.argmin(new_probs)

    return rep_options[chosen_ind], orig_prob - new_probs[chosen_ind]


def get_named_entity_replacements():
    imdb_0 = {'PERSON': 'Stewart',
                'NORP': 'European',
                'FAC': 'Classic',
                'ORG': 'ABC',
                'GPE': 'China',
                'LOC': 'South',
                'PRODUCT': 'Atlantis',
                'EVENT': 'Hugo',
                'WORK_OF_ART': 'Shower',
                'LAW': 'YouTube',
                'LANGUAGE': 'French',
                'DATE': 'Sunday',
                'TIME': 'hours',
                'PERCENT': '75-80%',
                'MONEY': '3',
                'QUANTITY': 'two-parter',
                'ORDINAL': 'fifth',
                'CARDINAL': '10/10',
              }
    # If the original input in IMDB belongs to class 1 (positive)
    imdb_1 = {'PERSON': 'Dracula',
                'NORP': 'Christian',
                'FAC': 'FX',
                'ORG': 'CGI',
                'GPE': 'Texas',
                'LOC': 'Dahmer',
                'PRODUCT': 'F',
                'EVENT': 'Olympics',
                'WORK_OF_ART': 'Dahmer',
                'LAW': 'DUMB',
                'LANGUAGE': 'Algerian',
                'DATE': '1970',
                'TIME': 'evening',
                'PERCENT': '80%',
                'MONEY': '10',
                'QUANTITY': '35mm',
                'ORDINAL': 'Secondly',
                'CARDINAL': '4',
              }
    imdb = [imdb_0, imdb_1]

    toxic_0 = {'PERSON': 'poop',
                'NORP': 'Nazi',
                'FAC': 'Pro-Assad',
                'ORG': 'HATE',
                'GPE': 'yourselfgo',
                'LOC': 'DELETE',
                'PRODUCT': 'SPICS',
                'EVENT': 'series',
                'WORK_OF_ART': 'Brown',
                'LAW': 'WISHES',
                'LANGUAGE': 'hebrew',
                'DATE': '1984',
                'TIME': 'seconds',
                'PERCENT': '98%',
                'MONEY': '578,254.00',
                'QUANTITY': '1-800-277-4653',
                'ORDINAL': 'FIRST',
                'CARDINAL': '248',
               }
    # If the original input in IMDB belongs to class 1 (positive)
    toxic_1 = {'PERSON': 'Jesus',
                'NORP': 'German',
                'FAC': 'Wikipedia',
                'ORG': 'POV',
                'GPE': 'U.S.',
                'LOC': 'West',
                'PRODUCT': 'Wikipedians',
                'EVENT': 'Holocaust',
                'WORK_OF_ART': 'cellpadding=""0',
                'LAW': 'Copyrights',
                'LANGUAGE': 'Mandarin',
                'DATE': '2005',
                'TIME': 'minutes',
                'PERCENT': 'width=""100%',
                'MONEY': '084080',
                'QUANTITY': '0.5mm',
                'ORDINAL': 'secondary',
                'CARDINAL': 'four',
               }
    toxic = [toxic_0, toxic_1]

    spam_0 = {'PERSON': 'nokia',
                'NORP': 'spanish',
                'FAC': '83600',
                'ORG': 'sony',
                'GPE': 'uk',
                'LOC': 'call2optout',
                'PRODUCT': 'dvd',
                'EVENT': 'olympics',
                'WORK_OF_ART': '11mths+',
                'LAW': '050703',
                'DATE': 'weekly',
                'TIME': 'midnight',
                'PERCENT': '50%',
                'MONEY': '5000',
                'QUANTITY': '40gb',
                'ORDINAL': '3ss',
                'CARDINAL': '16',
              }
    # If the original input in IMDB belongs to class 1 (positive)
    spam_1 = {'PERSON': 'lol',
                'NORP': 'chinese',
                'FAC': '2wks',
                'ORG': 'wat',
                'GPE': 'india',
                'LOC': 'earth',
                'PRODUCT': 'bmw',
                'WORK_OF_ART': 'want2come',
                'LANGUAGE': 'english',
                'DATE': 'yesterday',
                'TIME': 'tonight',
                'PERCENT': '20%',
                'MONEY': '#',
                'QUANTITY': '40mph',
                'ORDINAL': 'second',
                'CARDINAL': 'one',
              }
    spam = [spam_0, spam_1]

    clickbait_0 = {'PERSON': 'valentine',
                    'NORP': 'indian',
                    'ORG': 'google',
                    'GPE': 'mexico',
                    'LOC': 'west',
                    'PRODUCT': '|',
                    'DATE': "'90s",
                    'TIME': 'evening',
                    'PERCENT': '1%',
                    'MONEY': '30under30',
                    'QUANTITY': '7cm',
                    'ORDINAL': '6th',
                    'CARDINAL': '7',
                   }
    # If the original input in IMDB belongs to class 1 (positive)
    clickbait_1 = {'PERSON': 'donald',
                    'NORP': 'australian',
                    'FAC': 'metro',
                    'ORG': 'senate',
                    'GPE': 'florida',
                    'LOC': 'asia',
                    'PRODUCT': 'cole',
                    'EVENT': 'olympics',
                    'WORK_OF_ART': 'bible',
                    'LAW': 'roe',
                    'LANGUAGE': 'latin',
                    'DATE': 'sunday',
                    'TIME': 'obama',
                    'PERCENT': '30%',
                    'MONEY': 'hot100',
                    'ORDINAL': 'fourth',
                    'CARDINAL': 'two',
                   }

    clickbait = [clickbait_0, clickbait_1]

    return {'imdb': imdb, 'toxic': toxic, 'spam': spam, 'clickbait': clickbait}


def recognize_named_entity(texts):
    ne_freq_dict = copy.deepcopy(NE_type_dict)

    for text in texts:
        doc = nlp(text)
        for word in doc.ents:
            ne_freq_dict[word.label_][word.text] += 1
    return ne_freq_dict


def find_adv_ne(ne_true, ne_false):
    for ne_type in NE_type_dict.keys():
        # find the most frequent true and other NEs of the same type
        true_ne_list = [NE_tuple[0] for (i, NE_tuple) in enumerate(ne_true[ne_type]) if i < 15]
        other_ne_list = [NE_tuple[0] for (i, NE_tuple) in enumerate(ne_false[ne_type]) if i < 30]

        for other_NE in other_ne_list:
            if other_NE not in true_ne_list and len(other_NE.split()) == 1:
                print("'" + ne_type + "': '" + other_NE + "',")
                break


def find_adversarial_ne(path):
    df = pd.read_csv(path)

    pos_texts = df[df.label == 1].content
    neg_texts = df[df.label == 0].content

    # calculate word frequencies
    pos_ne_freq = recognize_named_entity(pos_texts)
    neg_ne_freq = recognize_named_entity(neg_texts)

    # replace dictionaries with ordered tuple lists
    for ne_type in NE_type_dict.keys():
        pos_ne_freq[ne_type] = sorted(pos_ne_freq[ne_type].items(), key=lambda k_v: k_v[1], reverse=True)
        neg_ne_freq[ne_type] = sorted(neg_ne_freq[ne_type].items(), key=lambda k_v: k_v[1], reverse=True)

    # find positive class and the negative
    print("now for class 1 !!!")
    find_adv_ne(pos_ne_freq, neg_ne_freq)

    print("now for class 0 !!!")
    find_adv_ne(neg_ne_freq, pos_ne_freq)


if __name__ == '__main__':
    find_adversarial_ne('../../../data/clickbait/clickbait_train_clean.csv')
