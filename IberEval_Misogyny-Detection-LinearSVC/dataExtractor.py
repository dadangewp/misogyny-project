# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 16:45:35 2018

@author: dadangewp
"""

import codecs
import re

def swearingWordsCount(tweetText):
    ironicWords = set(["fuck","fucking","fucked","shit","ass","arse","bitch","nigga","nigger","hell","whore","dick","pussy","piss","slit","puta","tit","tits","damn","fag","faggot","faggots","cunt","cum","blowjob","retard","bastard","wtf"])
    return len(tweetText.intersection(ironicWords))

def writeDataset(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    annotations = []
    texts = []
    swearWords = set(["fuck","shit","arse","bitch","nigga","nigger","hell","whore","dick","pussy","piss","slit","puta","tits","damn","fag","faggot","faggots","cunt","cum","blowjob","retard","bastard","wtf"])
    FNAME = './none-corpus.txt'
    PREDICTIONSFILE = codecs.open(FNAME, "w", "latin-1")
    with codecs.open(fp, encoding="latin-1") as data_in:
        for line in data_in:
            #line = line.rstrip() # remove trailing whitespace
            #print (line)
            annotation = line.split("\t")[1]
            text = line.split("\t")[2]
            print(annotation +" "+ text)
            count = 0
            for swear in swearWords:
                if swear in text.lower():
                    count = count + 1
            if(annotation == "none" and count > 0) :
                texts.append(text)
                annotations.append(annotation)
                PREDICTIONSFILE.write("none"+"\t"+text+"\n")
    PREDICTIONSFILE.close()        
    return annotations,texts


datapath = "D:\\PhD\\Discriminating Insulting and Not Insulting in Swearing Contained Tweet\\hatespeech-corpora\\hatespeech-master\\data\\twitter\\train-edited.txt"
annotations, texts = writeDataset(datapath)
print (len(annotations))
print (len(texts))
