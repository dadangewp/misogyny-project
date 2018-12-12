# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 20:58:43 2017

@author: dadangewp
"""

'''
description: implementation of feature selection, preprocessing and scaling
'''
from features import userBased, contentBased
from features import featureEvaluation
import sklearn.preprocessing as pp
from iodata.saveToFile import saveMatrixToCSVFile
from iodata.saveToFile import saveTweetToCSVFile
import numpy as np
import re
from collections import Counter
from scipy.sparse import hstack, csr_matrix

def featureExtraction(data,gt,model): 
    
    featureMatrix = []
    featurePrint = []
    labelMatrix = [] 
    tweetMatrix = []
    combinedMatrix = []
    features = [#"Tweet",
                #"negation",
                #"denywordcount",
                #"questMarkCount",
                #"en_an",
                #"en_asf",
                #"en_asm",
                #"en_cds",
                #"en_ddf",
                #"en_ddp",
                #"en_dmc",
                #"en_is",
                #"en_om",
                #"en_or",
                #"en_pa",
                #"en_pr",
                #"en_ps",
                #"en_qas",
                #"en_rci",
                #"en_re",
                #"en_svp",
                "swearWordCount",
                "swearWordPresence",
                #"repeatedChar",
                #"questMark",
                #"exclMarkCount",
                "hashTagsPresence",
                #"hashTagsCount",
                #"mentionCount",
                #"newLine",
                #"mentionFiltered",
                #"hasDotDot",
                #"numberDotDot",
                #"numberOfColon",
                #"hasQuote",
                #"ConjunctionCount",
                #"PrepCount",
                #"countPronoun",
                #"pronounPresence",
                #"upperCase",
                #"textLengh",
                "LinkCount",
                #"LinkPresence",
                #"emojiCount",
                #"emojiPresence",
                #"emoticonCount"]
                #"resolveURL",
                #"sharedLink",
                #"womenPronoun",
                "sexistSlurs",
                "womenWord"]
                #"manTarget",
                #"emojiSentiment",
                #"capitalWordCount"]
                #"countAdjective"]
                #"countNouns",
                #"interjectionWord",
                #"countVerbs"]
                #"Reply-To-Sim",
                #"TweetHierarchy",
                #"sentimentScore"]
    replyToTweetText = "-"
    clients = list()
    index = 0
    for tweet in data:
                #creating class labels
                label = gt[index]
                    
                
                #tweet console output
                '''
                print PPSourceTweetContent
                print PPTweetContent
                print tweetContent["source"]
                
                if label == "deny":
                    try:
                        print tweetContent["text"]
                    except:
                        print "error"
                        
                if(label=="deny"):
                        try:
                            print str(tweet)+": "+ sourceTweetContent["text"] + "::==::" + tweetContent["text"]
                        except:
                            print "print not possible"
                '''
                splittedTweet = set([]) 
                copiedTweet = tweet
                copiedTweet = re.sub(r'[^\w\s]','',copiedTweet)
                #print(copiedTweet)
                tweetToken = copiedTweet.split(" ")
                #print (tweetToken)
                #resolvedTweet = contentBased.resolveURL(copiedTweet)
                #print (resolvedTweet)
                for word in tweetToken:
                    word = word.lower()
                    #print(word)
                    splittedTweet.add(word)
                if label != "":
                #if label == "query":
                    labelMatrix.append(label)
                    tweetMatrix.append(tweet) 
                    #creating feature vector    
                    featureVector = ([#str(tweetContent["text"]),
                                          #contentBased.questionMarksCount(tweet),
                                          #contentBased.negationWordsCount(tweet),
                                          #contentBased.lexicon_en_an(tweet),
                                          #contentBased.lexicon_en_asf(tweet),
                                          #contentBased.lexicon_en_asm(tweet),
                                          #contentBased.lexicon_en_cds(tweet),
                                          #contentBased.lexicon_en_ddf(tweet),
                                          #contentBased.lexicon_en_ddp(tweet),
                                          #contentBased.lexicon_en_dmc(tweet),
                                          #contentBased.lexicon_en_is(tweet),
                                          #contentBased.lexicon_en_om(tweet),
                                          #contentBased.lexicon_en_or(tweet),
                                          #contentBased.lexicon_en_pa(tweet),
                                          #contentBased.lexicon_en_pr(tweet),
                                          #contentBased.lexicon_en_ps(tweet),
                                          #contentBased.lexicon_en_qas(tweet),
                                          #contentBased.lexicon_en_rci(tweet),
                                          #contentBased.lexicon_en_re(tweet),
                                          #contentBased.lexicon_en_svp(tweet),
                                          contentBased.getSwearCount(tweet),
                                          contentBased.getSwearPresence(tweet),
                                          #contentBased.repeatedChar(tweet),
                                          #contentBased.questionMark(tweet),
                                          #contentBased.exclamationMarks(tweet),
                                          contentBased.hashTagsPresence(tweet),
                                          #contentBased.hashTagsCount(tweet),
                                          #contentBased.mentionMarks(tweet),
                                          #contentBased.newLineMark(tweet),
                                          #contentBased.mentionMarksFiltered(tweet),
                                          #contentBased.hasDotDotDot(tweet),
                                          #contentBased.numberOfDotDotDot(tweet),
                                          #contentBased.colonCount(tweet),
                                          #contentBased.getQuoteCount(tweet),
                                          #contentBased.countConjunction(tweet),
                                          #contentBased.countPreposition(tweet),
                                          #contentBased.countProNoun(tweet),
                                          #contentBased.proNounPresence(tweet),
                                          #contentBased.upperCaseCount(tweet),
                                          #contentBased.textLenght(tweet),
                                          contentBased.linksCount(tweet),
                                          #contentBased.linkPresence(tweet),
                                          #contentBased.getEmojiPresence(tweet),
                                          #contentBased.getEmojiCount(tweet),
                                          #contentBased.emoticonCount(tweet)
                                          #contentBased.resolveURL(tweet),
                                          #contentBased.sharedLink(splittedTweet),
                                          #ontentBased.womenPronoun(tweet),
                                          contentBased.sexistSlurs(tweet),
                                          contentBased.ironicWordsCount(tweet)
                                          #contentBased.manTarget(tweet),
                                          #contentBased.getEmojiSentiment(tweet),
                                          #contentBased.capitalWordCount(tweet)
                                          #contentBased.countAdjective(tweet)
                                          #contentBased.countNoun(tweet),
                                          #contentBased.nounPresence(tweet)
                                          #contentBased.interjectionWord(splittedTweet),
                                          #contentBased.countVerbs(tweet)
                                          #contentBased.textSimToSource([PPReplyToTweetContent,PPTweetContent]),
                                          #contentBased.tweetLevel(sourceTweet,tweet,hierarchy),
                                          #contentBased.sentimentScore(tweet)
                                          #contentBased.sentimentSimilarity(sourceTweetContent["text"],tweetContent["text"])
                                        ])
                    #array = contentBased.addVector()
                    #for value in array:
                    #    featureVector.append(value)
                    featureMatrix.append(featureVector[:len(features)]) #number of features
                    featurePrint.append(tweet)
                    #print (featureVector)
                    #print("\n")
                    featureVector.append(label)
                    #featureVector.append(str(tweet))
                    combinedMatrix.append(featureVector)  
                    index = index + 1
 
    #print ("features:")                
    #print (features) 
    
    nrClients = Counter(clients)
                                                
    # standardization (zero mean, variance of one)
    stdScale = pp.StandardScaler().fit(featureMatrix)
    featureMatrixScaled = stdScale.transform(featureMatrix)
    
    #file output
    saveMatrixToCSVFile(featureMatrix,"featureMatrix.csv")
    saveTweetToCSVFile(labelMatrix,"tweetTextTrain.txt")
    #saveTweetToCSVFile(labelMatrix,"LabelMatrixTrain.txt")
    #print ("the size is :" + str(len(featurePrint)))
    #saveMatrixToCSVFile(featureMatrixScaled,"featureScaleMatrix.csv")
    #saveMatrixToCSVFile(labelMatrix,"labelMatrix.csv")
    #saveMatrixToCSVFile(combinedMatrix,"featureLabelMatrix.csv")
     
    #featureEvaluation.featureClassCoerr(featureMatrix,labelMatrix) 
    #print(features)
                              
    #return pp.normalize(featureMatrixScaled), labelMatrix, tweetMatrix                    
    return featureSelection(featureMatrixScaled), labelMatrix, tweetMatrix, features        

def featureSelection(featureMatrix):
    return featureMatrix