'''
description: implementations of content-based features
'''
from collections import Counter
import time
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import re
import requests
import numpy as np
import datetime as dt
from features.wvBased import WVBased
from nltk import word_tokenize, pos_tag
from nltk.parse import stanford
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from irony.emojiExtractor import Emoji
import emoji
from irony.emojiSentiment import EmojiSentiment
from irony.swearWordExtractor import Swear
from lexicon_en.en_an import EN_AN
from lexicon_en.en_asf import EN_ASF
from lexicon_en.en_asm import EN_ASM
from lexicon_en.en_cds import EN_CDS
from lexicon_en.en_ddf import EN_DDF
from lexicon_en.en_ddp import EN_DDP
from lexicon_en.en_dmc import EN_DMC
from lexicon_en.en_is import EN_IS
from lexicon_en.en_om import EN_OM
from lexicon_en.en_or import EN_OR
from lexicon_en.en_pa import EN_PA
from lexicon_en.en_pr import EN_PR
from lexicon_en.en_ps import EN_PS
from lexicon_en.en_qas import EN_QAS
from lexicon_en.en_rci import EN_RCI
from lexicon_en.en_re import EN_RE
from lexicon_en.en_svp import EN_SVP

parser = stanford.StanfordParser(model_path="D:\PhD\RumourEval\Small Project on Stance Detection in Rumour\stanford-parser-full-2017-06-09\model\englishPCFG.ser.gz")
#model = gensim.models.Word2Vec.load('brown_model')
sid = SentimentIntensityAnalyzer()
emojiex = Emoji()
wvbased = WVBased()
emosent = EmojiSentiment()
swear = Swear()
en_an = EN_AN()
en_asf = EN_ASF()
en_asm = EN_ASM()
en_cds = EN_CDS()
en_ddf = EN_DDF()
en_ddp = EN_DDP()
en_dmc = EN_DMC()
en_is = EN_IS()
en_om = EN_OM()
en_or = EN_OR()
en_pa = EN_PA()
en_ps = EN_PS()
en_pr = EN_PR()
en_qas = EN_QAS()
en_rci = EN_RCI()
en_re = EN_RE()
en_svp = EN_SVP()

def textSimToSource(tweetTexts):
      
    jaccard = float(len(tweetTexts[0].intersection(tweetTexts[1]))/float(len(tweetTexts[0].union(tweetTexts[0])))) 
    
    # count word occurrences
    #a_vals = Counter(tweetTexts[0])
    #b_vals = Counter(tweetTexts[1])

    # convert to word-vectors
    #words  = list(a_vals.keys() | b_vals.keys())
    #a_vect = [a_vals.get(word, 0) for word in words]        # [0, 0, 1, 1, 2, 1]
    #b_vect = [b_vals.get(word, 0) for word in words]        # [1, 1, 1, 0, 1, 0]

    # find cosine
    #len_a  = sum(av*av for av in a_vect) ** 0.5             # sqrt(7)
    #len_b  = sum(bv*bv for bv in b_vect) ** 0.5             # sqrt(4)
    #dot    = sum(av*bv for av,bv in zip(a_vect, b_vect))    # 3
    #cosine = dot / (len_a * len_b)  
    return jaccard

def getEmojiCount(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = emojiex.getEmojiCount(cleanedTweet)
    return score

def getEmojiPresence(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    cleanedTweet = emoji.demojize(cleanedTweet)
    score = emosent.get_emoji_sentiment(cleanedTweet)
    #print (score)
    if(score != 0):
        return 1
    else :
        return 0

def getEmojiSentiment(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet = emoji.demojize(cleanedTweet)
    cleanedTweet.lower()
    score = emosent.get_emoji_sentiment(cleanedTweet)
    if score > 0:
        return 1
    elif score < 0:
        return 2
    else:
        return 0
    #print (score)
    #if(score > 0):
    #    return 1
    #else :
    #    return 0

def getSwearCount(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = swear.getSwearCount(tweetText)
    return score

def getSwearPresence(tweetText):
    score = swear.getSwearCount(tweetText)
    if score > 0:
        return 1
    else :
        return 0

def repeatedChar(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    repeat = len(re.findall(r'((\w)\2{2,})', cleanedTweet))
    if repeat > 0:
        return 1
    else :
        return 0

def emojiIncongruity(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = getEmojiSentiment(tweetText)
    textsent = sentimentScore(cleanedTweet)
    distance = textsent - score
    if score != 0.0:
        #if textsent == 0.0:
        #    return 0
        if (score < 0 and textsent > 0) or (score > 0 and textsent < 0):
            return 1
        elif (score > 0 and textsent > 0) and (abs(score-textsent)>0.5):
            return 1
        elif (score < 0 and textsent < 0) and (abs(score-textsent)>0.5):
            return 1
        else:
            return 0
    else:
        return 0
    #if score == 0.0:
    #    return 0
    #else:
    #return distance

def retweetCount(tweet): 
    return tweet["retweet_count"]

def avgWordLength(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sentenceLength = len(cleanedTweet)
    tweetTokenList = tokenizer(cleanedTweet)
    wordCount = len(tweetTokenList)
    if (sentenceLength > 0 and wordCount > 0):
        avg = float(sentenceLength/wordCount)
        avg = format (avg,'.2f')
        return float(avg)
    else :
        return 0 

def countProNoun(tweetText):
    score = sum(1 for word, pos in pos_tag(word_tokenize(tweetText)) if pos.startswith('PRP'))
    return score

def proNounPresence(tweetText):
    if countProNoun(tweetText) > 0:
        return 1
    else :
        return 0

def countNoun(tweetText):
    score = sum(1 for word, pos in pos_tag(word_tokenize(tweetText)) if pos.startswith('NN'))
    return score

def nounPresence(tweetText):
    if countNoun(tweetText) > 0:
        return 1
    else :
        return 0

def countAdjective(tweetText):
    score = sum(1 for word, pos in pos_tag(word_tokenize(tweetText)) if pos.startswith('JJ'))
    return score

def countVerbs(tweetText):
    score = sum(1 for word, pos in pos_tag(word_tokenize(tweetText)) if pos.startswith('V'))
    return score

def countConjunction(tweetText):
    score = sum(1 for word, pos in pos_tag(word_tokenize(tweetText)) if pos.startswith('IN'))
    return score

def countPreposition(tweetText):
    score = sum(1 for word, pos in pos_tag(word_tokenize(tweetText)) if pos.startswith('CC'))
    return score

def supportWordsCount(tweetText):
    SUPPORTWORDS = set(["clarifi", "right", "evid", "confirm", "support", "definit", "discov", "explain", "truth", "true", "offici"])
    #tweetTokenList = tokenizer(tweetText) 
    #return len([token for token in SUPPORTWORDS if token in tweetTokenList])
    return len(tweetText.intersection(SUPPORTWORDS))

def commentWordsCount(tweetText):
    COMMENTWORDS = set(["comment", "claim", "accord", "sourc", "show", "captur", "say", "report", "observ", "footag"])
    return len(tweetText.intersection(COMMENTWORDS))

def emoticonCount(tweetText):
    emoticons = [":)",":p",":P"]
    count = 0
    for emot in emoticons:
        if emot in tweetText:
            count = count + 1
    return count

def capitalWordCount(tweetText):
    num = 0
    for char in tweetText:
        if char.isupper():
            num+=1
    #print (num)
    return num

def lexicon_en_an(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = en_an.get_en_an_count(cleanedTweet)
    #return score
    if score > 0:
        return 1
    else :
        return 0

def lexicon_en_asf(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = en_asf.get_en_asf_count(cleanedTweet)
    #return score
    if score > 0:
        return 1
    else :
        return 0

def lexicon_en_asm(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = en_asm.get_en_asm_count(cleanedTweet)
    #return score
    if score > 0:
        return 1
    else :
        return 0

def lexicon_en_cds(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = en_cds.get_en_cds_count(cleanedTweet)
    #return score
    if score > 0:
        return 1
    else :
        return 0

def lexicon_en_ddf(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = en_ddf.get_en_ddf_count(cleanedTweet)
    #return score
    if score > 0:
        return 1
    else :
        return 0

def lexicon_en_ddp(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = en_ddp.get_en_ddp_count(cleanedTweet)
    #return score
    if score > 0:
        return 1
    else :
        return 0

def lexicon_en_dmc(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = en_dmc.get_en_dmc_count(cleanedTweet)
    #return score
    if score > 0:
        return 1
    else :
        return 0

def lexicon_en_is(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = en_is.get_en_is_count(cleanedTweet)
    #return score
    if score > 0:
        return 1
    else :
        return 0

def lexicon_en_om(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = en_om.get_en_om_count(cleanedTweet)
    #return score
    if score > 0:
        return 1
    else :
        return 0

def lexicon_en_or(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = en_or.get_en_or_count(cleanedTweet)
    #return score
    if score > 0:
        return 1
    else :
        return 0

def lexicon_en_pa(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = en_pa.get_en_pa_count(cleanedTweet)
    #return score
    if score > 0:
        return 1
    else :
        return 0

def lexicon_en_pr(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = en_pr.get_en_pr_count(cleanedTweet)
    #return score
    if score > 0:
        return 1
    else :
        return 0

def lexicon_en_ps(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = en_ps.get_en_ps_count(cleanedTweet)
    #return score
    if score > 0:
        return 1
    else :
        return 0

def lexicon_en_qas(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = en_qas.get_en_qas_count(cleanedTweet)
    #return score
    if score > 0:
        return 1
    else :
        return 0

def lexicon_en_rci(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = en_rci.get_en_rci_count(cleanedTweet)
    #return score
    if score > 0:
        return 1
    else :
        return 0

def lexicon_en_re(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = en_re.get_en_re_count(cleanedTweet)
    #return score
    if score > 0:
        return 1
    else :
        return 0

def lexicon_en_svp(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = en_svp.get_en_svp_count(cleanedTweet)
    #return score
    if score > 0:
        return 1
    else :
        return 0

def ironicWordsCount(tweetText):
    tweet = tweetText.lower()
    ironicWords = set(["women","woman","girl","lady","ladi","female","feminine","wife","housewife","feme","femini"])
    count = 0
    for i in ironicWords:
        score = len(re.findall(i, tweet))
        count = count + score
    if count > 0:
        return 1
    else :
        return 0

def sexistSlurs(tweetText):
    tweet = tweetText.lower()
    ironicWords = set(["ugly","foxy","cunt","floozy","doll","pussy","chick","babe","bitch","tramp","whore","slut","showgirl","vagina","rape","rapis","fugly","slag","skank"])
    count = 0
    for i in ironicWords:
        score = len(re.findall(i, tweet))
        count = count + score
    if count > 0:
        return 1
    else :
        return 0

def manTarget(tweetText):
    tweet = tweetText.lower()
    ironicWords = set(["donald","trump","barack"])
    count = 0
    for i in ironicWords:
        score = len(re.findall(i, tweet))
        count = count + score
    if count > 0:
        return 1
    else :
        return 0

def womenPronoun(tweetText):
    tweet = tweetText.lower()
    ironicWords = set(["she","her"])
    count = 0
    for i in ironicWords:
        score = len(re.findall(i, tweet))
        count = count + score
    if count > 0:
        return 1
    else :
        return 0


def sharedLink(tweetText):
    sharedLink = set(["via","visit"])
    return len(tweetText.intersection(sharedLink))

def interjectionWord(tweetText):
    interjects = set(["uh","oh","yeah"])
    return len(tweetText.intersection(interjects))

def resolveURL(tweetText):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweetText)
    #print(urls)
    flag = 0
    for url in urls:
        link = ""
        try:
            link = requests.get(url, verify=False, timeout=10).url
        except :
            #print ("connection refused")
            link = "error"     
        link = str(link)
        #print (link)
        if "suspended" in link:
            flag = 1
        else :
            continue
    
    #print("hasil = "+ str(flag))
    if flag == 1:
        return 1
    else :
        return 0

def denyWordsPresence(text):
    SwearWords = ["smh","bs","li","lie","ly","lol","bullshit","wtf","fuck","idiot","aw"]
    
    count = len(text.intersection(SwearWords)) 

    if (count > 0):
        return 1
    else:
        return 0

def denyWordsCount(text):
    SwearWords = ["smh","bs","li","lie","ly","lol","bullshit","wtf","fuck","idiot","aw"]
    
    count = len(text.intersection(SwearWords)) 

    return count

def questionWordsCount(tweetText):
    QUESTIONWORDS = set(["what", "who", "question", "whi", "how", "where", "wonder"])
    return len(tweetText.intersection(QUESTIONWORDS))

def questionMarksCount(tweetText):
    return len(re.findall("\?", tweetText))

def colonCount(tweetText):
    count = len(re.findall(":", tweetText))
    if (count > 0):
        return 1
    else :
        return 0
    
def upperCaseCount(tweetText):
    count = 0
    for i in tweetText:
        if i.isupper():
            count = count + 1
    #print (count)
    if count > 0:
        return 1
    else : 
        return 0

def questionMark(tweetText):
    count = len(re.findall("\?", tweetText))
    if (count > 0):
        return 1
    else:
        return 0

def newLineMark(tweetText):
    count = len(re.findall("\|", tweetText))
    if (count > 0):
        return 1
    else:
        return 0
    
def exclamationMarks(tweetText):
    count = len(re.findall("\!\!\!", tweetText))
    if count > 0:
        return 1
    else:
        return 0

def mentionMarks(tweetText):
    count = len(re.findall("\@", tweetText))
    if count == 0:
        return 0
    else :
        return 1

def mentionMarksFiltered(tweetText):
    count = len(re.findall("\@", tweetText))
    if count == 0:
        if len(tweetText) < 100 or len(tweetText) > 125:
            return 0
        else:
            return 1
    else :
        return 0

def getQuoteCount(tweetText):
    matches1 = len(re.findall(r'\"(.+?)\"',tweetText))
    matches2 = len(re.findall(r'\'(.+?)\'',tweetText))
    if (matches1 > 0 or matches2 > 0):
        return 1
    else:
        return 0

def hasDotDotDot(tweetText):
    count = len(re.findall("...", tweetText))
    if count > 0:
        return 1
    else:
        return 0

def numberOfDotDotDot(tweetText):
    count = len(re.findall("\.", tweetText))
    if count > 0:
        return 1
    else:
        return 0

def textLenght(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweetText).split())
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    tweetTokenList = tokenizer(cleanedTweet) 
    return len(tweetTokenList)

def linksCount(tweetText):
    return len(re.findall("http", tweetText))

def linkPresence(tweetText):
    if(len(re.findall("http", tweetText)) > 0):
        return 1
    else:
        return 0
    
def hashTagsCount(tweetText):
    count = len(re.findall("#", tweetText))
    if count == 0:
        return 0
    elif count > 3:
        return 2
    else:
        return 1
    #return count

def hashTagsPresence(tweetText):
    if(len(re.findall("#", tweetText)) > 0):
        return 1
    else:
        return 0
    
def swearWords(text):
    SwearWords = ["fuck","asshole","shit","ass","bitch","nigga","hell","whore","dick","pussy","slut","putta","damn","fag","cum","cunt","cock","blowjob","retard"]
    count = 0

    for w in SwearWords:
        count += len(re.findall(w, text))

    return count

def bowTokenizer(text):
    #nltk.download("stopwords")
    stopTerms = set(stopwords.words("english"))
    tokens = set(text.split()).difference(stopTerms)  
    return tokens

def tokenizer(text):   
    vectorizer = CountVectorizer(min_df=1)
    analyze = vectorizer.build_analyzer()
    return analyze(text)

def negationWordsCount(text):   
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",text).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sentences = parser.raw_parse(cleanedTweet)
    
    found = 0;
    #print (sentences)
    for line in sentences:
        for sentence in line:
            words = str (sentence)
            #print (words)
            if("neg" in words):
                found = 1
                #print ("found it")
                break

    if found == 1:
        return 1
    else :
        return 0
    
def stemmer(terms):
    
    stemmer = PorterStemmer()
    stemmedTerms = set([]) 
     
    try: 
        for term in terms:
            stemmedTerm = stemmer.stem(term)
            stemmedTerms.add(str(stemmedTerm))
        
        return stemmedTerms
    except:
        return terms
    
def replyTimeToSource(sourceDate,replyDate):
    
    sourceTime = dt.datetime.strptime(sourceDate[:sourceDate.__len__()-11],'%a %b %d %H:%M:%S')
    replyTime = dt.datetime.strptime(replyDate[:replyDate.__len__()-11],'%a %b %d %H:%M:%S')
    timeDelta = replyTime - sourceTime
    replyTime = (timeDelta.days * 24 * 60) + (timeDelta.seconds/60)
    replyTime = format (replyTime,'.2f')
    replyTime = float(replyTime)
    return replyTime

    
def possiblySensitive(value):
    
    if value == "True": # no boolean -> text
        return 1
    else:
        return 0
    
def twitterClient(link):
    
    if link.find("Twitter Web Client") != -1: 
        return 1
    if link.find("Twitter for Mac") != -1: 
        return 1
    if link.find("Twitter for Websites") != -1: 
        return 1
    if link.find("Twitter for ") != -1: 
        return 0
    else:
        return 0
    
def tweetLevel(sourceId,targetId,hierarchy):
    
    count = 0
    tweethierachy = ""

    if sourceId == targetId:
        return 0
    else:
        for i, t in hierarchy.items():
            if int(i) == sourceId:
                tweethierachy = str(t).replace("u", " ").split()
                break
    
        for x in tweethierachy:
            if (x.find("{") != -1):
                count = count + x.count('{')
            if (x.find("}") != -1):
                count = count - x.count('}')
            if (x.find(str(targetId)) != -1):
                return count
    
    return 0
            
def isQuestion(tweetText):
    #print ("masuk")
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    listSentence = []
    indexDelimiter = 0;
    indexDelimiter = cleanedTweet.find("?")
    if indexDelimiter != 0:
        cleanedTweet = cleanedTweet[:indexDelimiter+1] +"|"+cleanedTweet[indexDelimiter+1:]
    listTweet = cleanedTweet.split("|")
    for tweet in listTweet:
        splitTweet = tweet.split(".")
        for t in splitTweet:
            listSentence.append(t);
    sentences = parser.raw_parse_sents(listSentence)
    
    found = 0;
    #print (sentences)
    for line in sentences:
        for sentence in line:
            words = str (sentence)
            #print (words)
            if("SQ" in words or "SBARQ" in words):
                found = 1
                #print ("found it")
                break

    if found == 1:
        return 1
    else :
        return 0

def sentimentScore(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweetText).split())
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(cleanedTweet)
    if ss["compound"] > 0.0:
        return 1
    else :
        return 0
    
def avg_feature_vector(tweetText):
    #function to average all words vectors in a given paragraph 
    featureVec = np.zeros((100,), dtype="float32")
    nwords = 0
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweetText).split())
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    words = tokenizer(cleanedTweet)
    for word in words:
        featureVec = np.add(featureVec, model[word])
        nwords = nwords+1
    if(nwords>0):
        featureVec = np.divide(featureVec, nwords)
    return featureVec

def addVector():
    vector = np.full((3,1),7)
    return vector

def sentimentSimilarity(source,reply):
    cleanedTweet1 = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",source).split())
    cleanedTweet1.replace("'","")
    cleanedTweet1.replace('"',"")
    cleanedTweet1.replace('/',"")
    cleanedTweet1.replace("\\","")
    cleanedTweet2 = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",reply).split())
    cleanedTweet2.replace("'","")
    cleanedTweet2.replace('"',"")
    cleanedTweet2.replace('/',"")
    cleanedTweet2.replace("\\","")
    sid = SentimentIntensityAnalyzer()
    ss1 = sid.polarity_scores(cleanedTweet1)
    ss2 = sid.polarity_scores(cleanedTweet2)
    if (ss1 == ss2):
        return 1
    else :
        return 0

def unweightMaxSim(tweet,model):
    a,b,c,d,e,f,g,h = wvbased.get_wv_features(tweet,model)
    return a

def unweightMinSim(tweet,model):
    a,b,c,d,e,f,g,h = wvbased.get_wv_features(tweet,model)
    return b

def unweightMaxDis(tweet,model):
    a,b,c,d,e,f,g,h = wvbased.get_wv_features(tweet,model)
    return c

def unweightMinDis(tweet,model):
    a,b,c,d,e,f,g,h = wvbased.get_wv_features(tweet,model)
    return d

def weightMaxSim(tweet,model):
    a,b,c,d,e,f,g,h = wvbased.get_wv_features(tweet,model)
    return e

def weightMinSim(tweet,model):
    a,b,c,d,e,f,g,h = wvbased.get_wv_features(tweet,model)
    return f

def weightMaxDis(tweet,model):
    a,b,c,d,e,f,g,h = wvbased.get_wv_features(tweet,model)
    return g

def weightMinDis(tweet,model):
    a,b,c,d,e,f,g,h = wvbased.get_wv_features(tweet,model)
    return h