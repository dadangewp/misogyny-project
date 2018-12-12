# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:20:13 2018

@author: dadangewp
"""

from nltk import word_tokenize, pos_tag
import re


tweetText = "@SenSanders Sanders you cuckhold bitch. That military keeps your cunt ass free."
cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
cleanedTweet.replace("'","")
cleanedTweet.replace('"',"")
cleanedTweet.replace('/',"")
cleanedTweet.replace("\\","")
cleanedTweet.lower()
print(cleanedTweet)
tagged = pos_tag(word_tokenize(cleanedTweet))
#print (tagged)
#print (token)
combined_tag = ""
a = 0
for i in tagged:
    combined_tag = combined_tag +" "+str(tagged[a][1])
    a = a+1

print (combined_tag)