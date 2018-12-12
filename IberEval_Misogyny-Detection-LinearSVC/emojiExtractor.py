# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 20:07:08 2017

@author: dadangewp
"""

import codecs
import re
from nltk.stem.porter import PorterStemmer

class Emoji(object):

    emoji=[]

    def __init__(self):
        self.emoji = []
        #http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010
        file=codecs.open('D:/PhD/RumourEval/Small Project on Stance Detection in Rumour/affectiveResources/emojisentiment.txt', encoding='UTF-8')
        for line in file:
            word, score = line.strip().split(' ')
            word = str(word)
            self.emoji.append(word.lower())
        #print(self.liwcpos)    
        self.pattern_split = re.compile(r"\W+")
        return

    def getEmojiCount(self,text):
        
        counter=0
        #print (words)
        #print (self.liwcpos)
        for word in self.emoji:
            count = len(re.findall(word, text))
            counter = counter + count


        return counter
    
    def getEmojiList(self, text):
        emojis = ""
        for word in self.emoji:
            count = len(re.findall(word, text))
            if(count > 0):
                emojis = emojis +" "+word
        #print (emojis)
        return emojis


if __name__ == '__main__':
    emoji = Emoji()
    emojicount=emoji.getEmojiList("I enjoy stealing my husband hat every once in awhile :smiling_face_with_heart-shaped_eyes: :face_throwing_a_kiss: :flushed_face: #heyyall #GoodMorning http://t.co/LknoOm7VSb")
    print(emojicount)