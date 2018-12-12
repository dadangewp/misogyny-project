# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 19:06:37 2018

@author: dadangewp
"""

import tweepy
import codecs

def parse_dataset(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    ids = []
    annotations = []
    with codecs.open(fp, encoding="utf8") as data_in:
        for line in data_in:
            line = line.rstrip() # remove trailing whitespace
            idi = line.split(",")[0]
            annotation = line.split(",")[1]
            ids.append(idi)
            annotations.append(annotation)

    return ids,annotation
FNAME = './corpus.txt'
PREDICTIONSFILE = codecs.open(FNAME, "w", "utf-8")
consumer_key = 'SfB30I1x4cJxOVDpbydf5OxAn'
consumer_secret = 'Fp3XaeM59kGygFvryfV4wu0ROqclBKQ7din5d0JxR8h13iWGsP'
access_token = '911963260005552128-w9CwezEAPRbP15SVS8spZtyjg4einU2'
access_token_secret = 'PmWkfTFCR6p463AUNkFSfFcSTUAv61WJCLr481gYrto27'

datapath = "D:\\PhD\\Discriminating Insulting and Not Insulting in Swearing Contained Tweet\\hatespeech-master\\NAACL_SRW_2016.csv"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
ids, annotations = parse_dataset(datapath)
for idi in ids : 
    try :
        tweet = api.get_status(idi)
        PREDICTIONSFILE.write("{}\n".format(tweet.text))
        print(tweet.text)
    except:
        PREDICTIONSFILE.write("{}\n".format("deleted"))
        print("deleted")
PREDICTIONSFILE.close()