# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:35:56 2017

@author: dadangewp
"""

import gensim

print ("load wordvector")
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
word2vec = dict(zip(model.wv.index2word, model.wv.syn0))
tweet = ""