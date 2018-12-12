# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 20:50:56 2017

@author: dadangewp
"""

import codecs

def parse_training(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    y = []
    corpus = []
    with codecs.open(fp, encoding="utf-8") as data_in:
        for line in data_in:
            if not line.lower().startswith("id"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[1]
                misogyny = line.split("\t")[3]
                y.append(misogyny)
                corpus.append(tweet)

    return corpus, y

def parse_testing(fp):
    corpus = []
    y = []
    with codecs.open(fp, encoding="utf-8") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[1]
                misogyny = line.split("\t")[2]
                corpus.append(tweet)
                y.append(misogyny)

    return corpus,y