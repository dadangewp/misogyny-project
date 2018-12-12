# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 20:29:55 2017

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
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[1]
                misogyny = line.split("\t")[3]
                y.append(misogyny)
                corpus.append(tweet)

    return corpus, y

def parse_testing(fp):
    corpus = []
    with codecs.open(fp, encoding="utf-8") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[1]
                corpus.append(tweet)

    return corpus

def parse_label(fp):
    label = []
    with codecs.open(fp, encoding="utf-8") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                y = int(line)
                #if y == 0:
                 #   x = "not irony"
                #elif y == 1:
                 #   x = "polarity contrast"
                #elif y == 2 :
                 #   x = "other irony"
                #else :
                 #   x = "situational irony"
                label.append(y)

    return label

def parse_gold(fp):
    label = []
    with codecs.open(fp, encoding="utf-8") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                y = int(line.split("\t")[1])
                #if y == 0:
                 #   x = "not irony"
                #elif y == 1:
                 #   x = "polarity contrast"
                #elif y == 2 :
                 #   x = "other irony"
                #else :
                 #   x = "situational irony"
                label.append(y)

    return label