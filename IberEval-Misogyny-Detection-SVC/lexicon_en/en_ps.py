# -*- coding: utf-8 -*-
"""
Created on Tue May  8 03:21:13 2018

@author: dadangewp
"""

import codecs
import re
from nltk.stem.porter import PorterStemmer

class EN_PS(object):

    en_ps=[]

    def __init__(self):
        self.en_ps = []
        stemmer = PorterStemmer()
        #http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010
        file=codecs.open('D:\\PhD\\Misogyny Detection\\swear-word\\en_ps.txt', encoding='UTF-8')
        for line in file:
            word = line.strip("\r\n")
            word = stemmer.stem(word)
            self.en_ps.append(word)
        #print(self.liwcpos)    
        self.pattern_split = re.compile(r"\W+")
        return

    def get_en_ps_count(self,text):
        
        stemmer = PorterStemmer()
        counter=0
        words = self.pattern_split.split(text.lower())
        words = text.split(" ")
        for word in words:
            stemmed = stemmer.stem(word)
            if stemmed in self.en_ps:
                counter = counter + 1


        return counter


if __name__ == '__main__':
    en_ps = EN_PS()
    sentiment=en_ps.get_en_ps_count("fuck boob pussy")
    print(sentiment)