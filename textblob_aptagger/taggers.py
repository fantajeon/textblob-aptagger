# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import random
from collections import defaultdict
import pickle
import logging
import copy

from textblob.base import BaseTagger
from textblob.tokenizers import WordTokenizer, SentenceTokenizer
from textblob.exceptions import MissingCorpusError
from textblob_aptagger._perceptron import AveragedPerceptron

PICKLE = "trontagger-0.1.0.pickle"


class PerceptronTagger(BaseTagger):

    '''Greedy Averaged Perceptron tagger, as implemented by Matthew Honnibal.

    See more implementation details here:
        http://honnibal.wordpress.com/2013/09/11/a-good-part-of-speechpos-tagger-in-about-200-lines-of-python/

    :param load: Load the pickled model upon instantiation.
    '''

    START = ['-START-', '-START2-', '-START3-']
    END = ['-END-', '-END2-', '-END3-']
    AP_MODEL_LOC = os.path.join(os.path.dirname(__file__), PICKLE)

    def __init__(self, load=True):
        self.model = AveragedPerceptron()
        self.tagdict = {}
        self.classes = set()
        if load:
            self.load(self.AP_MODEL_LOC)

    def tag(self, corpus, tokenize=True):
        '''Tags a string `corpus`.'''
        # Assume untokenized corpus has \n between sentences and ' ' between words
        s_split = SentenceTokenizer().tokenize if tokenize else lambda t: t.split('\n')
        w_split = WordTokenizer().tokenize if tokenize else lambda s: s.split()
        def split_sents(corpus):
            for s in s_split(corpus):
                yield w_split(s)

        prev, prev2 = self.START
        tokens = []
        for words in split_sents(corpus):
            #context = self.START + [self._normalize(w) for w in words] + self.END
            context = self.START + [w for w in words] + self.END
            for i, word in enumerate(words):
                tag = self.tagdict.get(word)
                if not tag:
                    features = self._get_features(i, word, context, prev, prev2)
                    tag = self.model.predict(features)
                tokens.append((word, tag))
                prev2 = prev
                prev = tag
        return tokens

    def all_unk(self, tags):
        for t in tags:
            if not t is "UNK":
                return False
        return True


    def train(self, sentences, save_loc=None, nr_iter=5):
        '''Train a model from sentences, and save it at ``save_loc``. ``nr_iter``
        controls the number of Perceptron training iterations.

        :param sentences: A list of (words, tags) tuples.
        :param save_loc: If not ``None``, saves a pickled model in this location.
        :param nr_iter: Number of training iterations.
        '''
        self._make_tagdict(sentences)
        self.model.classes = self.classes
        for iter_ in range(nr_iter):
            c = 0
            n = 0
            not_unk = 0
            not_unk_c = 0
            for words, tags in sentences:
                #if self.all_unk(tags):
                #    if not random.randint(0,9) == 0:
                #        continue
                prev, prev2, prev3 = self.START
                #context = self.START + [self._normalize(w) for w in words] \
                context = self.START + [w for w in words] + self.END
                for i, word in enumerate(words):
                    if tags[i] == "UNK":
                        if not random.randint(0,9) == 0:
                            continue
                    #guess = self.tagdict.get(word)
                    #if not guess:
                    feats = self._get_features(i, word, context, prev, prev2, prev3)
                    guess = self.model.predict(feats)
                    self.model.update(tags[i], guess, feats)
                    prev3 = prev2
                    prev2 = prev
                    prev = guess
                    c += guess == tags[i]
                    n += 1
                    if not tags[i] is "UNK":
                        not_unk_c += guess == tags[i]
                        not_unk += 1
            random.shuffle(sentences)
            #logging.info("Iter {0}: {1}/{2}={3}".format(iter_, c, n, _pc(c, n)))
            print("Iter {0}: {1}/{2}={3}, {4}/{5}={6}".format(iter_, c, n, _pc(c, n), not_unk_c, not_unk, _pc(not_unk_c,not_unk) ))

            # Pickle as a binary file
            if save_loc is not None:
                backw = copy.deepcopy(self.model.weights)
                self.model.average_weights()
                modle_name = os.path.join(save_loc, "_{}.pkl".format(iter_))
                pickle.dump((self.model.weights, self.tagdict, self.classes), open(model_name, 'wb'), -1)
                self.model.weights = backw

        self.model.average_weights()
        modle_name = os.path.join(save_loc, "_final.pkl".format(iter_))
        pickle.dump((self.model.weights, self.tagdict, self.classes), open(save_loc, 'wb'), -1)
        return None

    def load(self, loc):
        '''Load a pickled model.'''
        try:
            w_td_c = pickle.load(open(loc, 'rb'))
        except IOError:
            msg = ("Missing trontagger.pickle file.")
            raise MissingCorpusError(msg)
        self.model.weights, self.tagdict, self.classes = w_td_c
        self.model.classes = self.classes
        return None

    def _normalize(self, word):
        '''Normalization used in pre-processing.

        - All words are lower cased
        - Digits in the range 1800-2100 are represented as !YEAR;
        - Other digits are represented as !DIGITS

        :rtype: str
        '''
        if '-' in word and word[0] != '-':
            return '!HYPHEN'
        elif word.isdigit() and len(word) == 4:
            return '!YEAR'
        elif word[0].isdigit():
            return '!DIGITS'
        else:
            return word.lower()

    def _get_features(self, i, word, context, prev, prev2, prev3):
        '''Map tokens into a feature representation, implemented as a
        {hashable: float} dict. If the features change, a new model must be
        trained.
        '''
        def add(name, *args):
            features[' '.join((name,) + tuple(args))] += 1

        i += len(self.START)
        features = defaultdict(int)
        # It's useful to have a constant feature, which acts sort of like a prior
        add('bias')
        add('i suffix', word[-3:])
        add('i pref1', word[0])
        add('i-1 tag', prev)
        add('i-2 tag', prev2)
        add('i-3 tag', prev3)
        add('i tag+i-2 tag', prev, prev2)
        add('i word', context[i])
        add('i-1 tag+i word', prev, context[i])
        add('i-1 word', context[i-1])
        add('i-1 suffix', context[i-1][-3:])
        add('i-2 word', context[i-2])
        add('i+1 word', context[i+1])
        add('i+1 suffix', context[i+1][-3:])
        add('i+2 word', context[i+2])
        return features

    def _make_tagdict(self, sentences):
        '''Make a tag dictionary for single-tag words.'''
        counts = defaultdict(lambda: defaultdict(int))
        for words, tags in sentences:
            for word, tag in zip(words, tags):
                counts[word][tag] += 1
                self.classes.add(tag)
        freq_thresh = 20
        ambiguity_thresh = 0.97
        for word, tag_freqs in counts.items():
            tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
            n = sum(tag_freqs.values())
            # Don't add rare words to the tag dictionary
            # Only add quite unambiguous words
            if n >= freq_thresh and (float(mode) / n) >= ambiguity_thresh:
                self.tagdict[word] = tag


def _pc(n, d):
    return (float(n) / d) * 100
