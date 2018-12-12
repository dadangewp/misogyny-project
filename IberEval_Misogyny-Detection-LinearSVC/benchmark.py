# -*- coding: utf-8 -*-
"""

@author: dadangewp
"""

from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import logging
import codecs


logging.basicConfig(level=logging.INFO)


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
                misogyny = line.split("\t")[2]
                y.append(misogyny)
                corpus.append(tweet)

    return corpus, y

def featurize(corpus):
    '''
    Tokenizes and creates TF-IDF BoW vectors.
    :param corpus: A list of strings each string representing document.
    :return: X: A sparse csr matrix of TFIDF-weigted ngram counts.
    '''
    
    tfidfVectorizer = CountVectorizer(ngram_range=(1,3),
                                              analyzer = "word",
                                          stop_words="english",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)
    feature  = []
    for tweet in corpus:
        feature.append(tweet)
    
    tfidfVectorizer = tfidfVectorizer.fit(feature)
    X_train = tfidfVectorizer.transform(feature)
    return X_train



if __name__ == "__main__":
    # Experiment settings

    # Dataset: AMI-IberEval 2018
    DIR_TRAIN = "D:\\PhD\\Hate Speech Evalita\\haspeede2018-master\\haspeede_FB-train.tsv"

    K_FOLDS = 10 # 10-fold crossvalidation
    #clf =  # the default, non-parameter optimized linear-kernel SVM
    clf = MLPClassifier(solver='adam',max_iter=400,hidden_layer_sizes=(25,2))  
    # Loading dataset and featurised simple Tfidf-BoW model
    corpus, y = parse_training(DIR_TRAIN)
    
    # Getting BoW feature
    X = featurize(corpus)
    
    # Training Classifier
    clf.fit(X,y)
    
    # Cross Validation
    scores = cross_val_score(clf, X, y, cv=K_FOLDS, scoring="accuracy")
    print(scores)
    print("Accuracy Score (Cross-V): %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    