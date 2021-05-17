# -*- coding: utf-8 -*-
"""
Spyder Editor

Running execute the grading assistant
"""

"""
Import necessary libaries
"""
import pandas as pd
import numpy as np
import re
import nltk

import weightedmedianfunc
import SVD_for_S

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import cohen_kappa_score

from scipy import sparse
###############

def StemmingWordList(arrayList):
    ps = nltk.PorterStemmer()
    newList = [ps.stem(word) for word in arrayList]
    return ''.join(newList)

svd = TruncatedSVD(n_iter=10, n_components=100)
tfidf = TfidfVectorizer(min_df = 0.01, max_df=0.85, stop_words='english')

#-----Import data-----#
train = pd.read_excel('./Data/training_set_rel3_set1.xlsx')
test = pd.read_excel('./Data/valid_set_set1.xlsx')
train.set_index('ID')
test.set_index('essay_id')
y_train = train['Score']
y_test = test['Score']
X = pd.concat([train,test])

train_numberOfSentences = X['Essay Content'].apply(lambda x: len(x.split('.')))
train_numberOfWords = X['Essay Content'].apply(lambda x: len(x.split()))
content = X['Essay Content']
content = content.apply(lambda x: re.sub('^[a-zA-Z]', ' ', x))
content = content.apply(lambda x: StemmingWordList(x))
x_transform = tfidf.fit_transform(content)
x_transform = sparse.hstack((x_transform, train_numberOfSentences[:,None]))
x_transform = sparse.hstack((x_transform, train_numberOfWords[:,None]))

# New SVD here
x_transform = svd.fit_transform(x_transform)

x_train = x_transform[:len(train)]
x_test = x_transform[len(train):]
neighbors = 6
nearestNeighbors = NearestNeighbors(n_neighbors=neighbors)
nearestNeighbors.fit(x_train)
test_dist, test_ind = nearestNeighbors.kneighbors(x_test)
"""
#----Using true median----#
prediction_list = list()
for val in test_ind:
    prediction_list.append(y_train[val[round(neighbors/2)]])
    
accuracy = cohen_kappa_score(y_test, prediction_list,weights='quadratic') 
print('True median', accuracy)

#----Using mean score----#
prediction_list = list()
for val in test_ind:
    total = 0
    for i in val:
        total += y_train[i]
    avg = round(total / len(val)) 
    prediction_list.append(avg)

accuracy = cohen_kappa_score(y_test, prediction_list,weights='quadratic') 
print('Using mean', accuracy)
"""
#---Using custom weighted----#
prediction_list = list()
n = len(test_ind)
for i in range(0, n):
    scores_list = list()
    dist_list = test_dist[i]
    for i in test_ind[i]:
        scores_list.append(y_train[i])
      
    prediction_list.append(round(weightedmedianfunc.weighted_median(scores_list,dist_list)))
          
accuracy = cohen_kappa_score(y_test, prediction_list,weights='quadratic') 
print('The accuracy of Using weighted median', accuracy)