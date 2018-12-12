# -*- coding: utf-8 -*-
"""
Created on Tue May  8 03:29:47 2018

@author: dadangewp
"""

import codecs
import re
from nltk.stem.porter import PorterStemmer

class EN_SVP(object):

    en_svp=[]

    def __init__(self):
        self.en_svp = []
        stemmer = PorterStemmer()
        #http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010
        file=codecs.open('D:\\PhD\\Misogyny Detection\\swear-word\\en_svp.txt', encoding='UTF-8')
        for line in file:
            word = line.strip("\r\n")
            word = stemmer.stem(word)
            self.en_svp.append(word)
        #print(self.liwcpos)    
        self.pattern_split = re.compile(r"\W+")
        return

    def get_en_svp_count(self,text):
        
        stemmer = PorterStemmer()
        counter=0
        words = self.pattern_split.split(text.lower())
        words = text.split(" ")
        for word in words:
            stemmed = stemmer.stem(word)
            if stemmed in self.en_svp:
                counter = counter + 1


        return counter


if __name__ == '__main__':
    en_svp = EN_SVP()
    sentiment=en_svp.get_en_svp_count("fuck boob pussy")
    print(sentiment)