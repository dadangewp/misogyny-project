# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 20:35:31 2017

@author: dadangewp
"""

from dataReader import parse_training
from dataReader import parse_testing
import configFeature as cfgFeature
from itertools import combinations
import featureManager
import gensim
import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
from sklearn import ensemble
from sklearn.model_selection import cross_val_score, cross_val_predict


DIR_TRAIN = "D:\\PhD\\Aditya Experiment Irony\\saif_dataset.txt"
#DIR_TEST = "D:\\PhD\\RumourEval\\Small Project on Irony Detection\\SemEval2018-Task3-master\\SemEval2018-Task3-master\\datasets\\test_TaskA\\SemEval2018-T3_input_test_taskA_annotated.txt"

if __name__ == '__main__':
    
    print ("started ...")
    #TASK = "A" # Define, A or B
    #FNAME = './predictions-task' + TASK + '.txt'
    #PREDICTIONSFILE = open(FNAME, "w")
    # read Training data
    print ("load wordvector")
    word2vec = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    #word2vec = dict(zip(model.wv.index2word, model.wv.syn0))
    print ("wordvector load finished")
    #word2vec = "zonk"
    feature_manager=featureManager.make_feature_manager()
    dataTrain, dataLabel = parse_training(DIR_TRAIN)
    #dataTest, labelTest = parse_testing(DIR_TEST)
    print ("Training data read") 
    feature_names=cfgFeature.feature_list['feature_names']
    #stuff = range(0, len(feature_names) )
    #parameters=[]
    #parameters_optimized=[]
    #highest=0
    #max_feature_set=[]
    #for L in range(1, len(stuff)+1):
        #for subset in combinations(stuff, L):
            #X_train,X_test=feature_manager.create_feature_space(dataTrain, word2vec, feature_names[list(subset)], train_tweets=None)
            #clf = svm.LinearSVC()
            #print ("training start")
            #clf.fit(X_train, dataLabel)
            #print ("training done")
            #scores = cross_val_score(clf, X_train, dataLabel, cv=10) 
            #acc = scores.mean()
            #predicted = cross_val_predict(clf,X_train,dataLabel,cv=10)
            #score = metrics.f1_score(dataLabel, predicted, pos_label=1)
            #print(feature_names[list(subset)])
            #print(score)
            #if score > 0.65:
                #print(str(feature_names[list(subset)]))
                #PREDICTIONSFILE.write(str(feature_names[list(subset)])+";")
                #print(score)
                #PREDICTIONSFILE.write(str(score))
                #PREDICTIONSFILE.write("\n")
    X_train = feature_manager.create_feature_space(dataTrain, word2vec, feature_names)
    clf = tree.ExtraTreeClassifier()
    clf.fit(X_train,dataLabel)
    #print (X_train.shape)
    print(cross_val_score(clf, X_train, dataLabel, cv=10, scoring="accuracy"))
    scores = cross_val_score(clf, X_train, dataLabel, cv=10, scoring="accuracy")
    print("F1 Score (Cross-V): %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #predicted = cross_val_predict(clf,X_train,dataLabel,cv=10)
    #predicted = clf.predict(X_test)
    #score = metrics.f1_score(labelTest, predicted, pos_label=1)
    #scoreTrain = metrics.f1_score(dataLabel, predictedTrain, pos_label=1)
    #print ("F1-score Task", TASK, score)
    #print (scoreTrain)
    #for p in predicted:
    #    PREDICTIONSFILE.write("{}\n".format(p))
    #PREDICTIONSFILE.close()
            