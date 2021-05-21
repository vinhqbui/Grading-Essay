# -*- coding: utf-8 -*-
"""
Spyder Editor

Running execute the grading assistant
"""

"""
Import necessary libaries
"""
import pandas as pd
import re
import nltk
import math

import weightedmedianfunc
import SVD_for_S

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 

import snowballstemmer

from scipy import sparse
###############
stop_words = set(stopwords.words('english')) 
def RemoveStopWords(arrayList):
    newList = [w for w in arrayList if not w in stop_words]
    return ' '.join(newList)


stemmer = nltk.LancasterStemmer()
def StemmingWordList(arrayList):
    newList = [stemmer.stem(word) for word in arrayList]
    return ' '.join(newList)

snowball = snowballstemmer.stemmer('english')
def SnowballStemmer(arrayList):
    words = snowball.stemWords(arrayList)
    return ' '.join(words)

wordNetLemmna = WordNetLemmatizer()
def WordNetLemma(arrayList):
    newList = [wordNetLemmna.lemmatize(word) for word in arrayList]
    return ' '.join(newList)
    
#-----Import data-----#
corpus = pd.read_excel('./Data/New_Corpus.xlsx')
Y = corpus['Score'].apply(lambda x: int(x))
X = corpus["Essay Content"]

train_numberOfSentences = X.apply(lambda x: len(x.split('.')))
train_numberOfWords = X.apply(lambda x: len(x.split()))
content = X
content = content.apply(lambda x: re.sub('[^a-zA-Z]+', ' ', x))
content = content.apply(lambda x: x.lower())

content = content.apply(lambda x: WordNetLemma(word_tokenize(x)))
content = content.apply(lambda x: RemoveStopWords(word_tokenize(x)))
content = content.apply(lambda x: SnowballStemmer(word_tokenize(x)))

dimensions = 100
neighbors = 5
#svd = TruncatedSVD(n_components=dimensions)
tfidf = TfidfVectorizer(min_df = 0.01, max_df=0.90, stop_words='english')

x_transform = tfidf.fit_transform(content)
x_transform = sparse.hstack((x_transform, train_numberOfSentences[:,None]))
x_transform = sparse.hstack((x_transform, train_numberOfWords[:,None]))

x_transform = SVD_for_S.SVD(x_transform.toarray(), dimensions)
#x_transform = svd.fit_transform(x_transform)


nearestNeighbors = NearestNeighbors(n_neighbors=neighbors+1)
nearestNeighbors.fit(x_transform)
test_dist, test_ind = nearestNeighbors.kneighbors(x_transform)


#Using regular median
prediction_list = list()
for item in test_ind:
    prediction_list.append(round((Y[item[math.floor(neighbors/2.0)+1]]+Y[item[math.ceil(neighbors/2.0)+1]])/2.0))
accuracy = cohen_kappa_score(Y, prediction_list,weights='quadratic') 
print('Using vanilla median', accuracy)

#----Using mean score----#
prediction_list = list()
for val in test_ind:
    total = 0
    for i in val[1:]:
        total += Y[i]
    avg = round(total / (len(val)-1)) 
    prediction_list.append(avg)

accuracy = cohen_kappa_score(Y, prediction_list,weights='quadratic') 
print('Using mean', accuracy)

#---Using custom weighted----#
prediction_list = list()
n = len(test_ind)
for i in range(0, n):
    scores_list = list()
    dist_list = test_dist[i][1:]
    for j in test_ind[i][1:]:
        scores_list.append(Y[j])      
    prediction_list.append(round(weightedmedianfunc.weighted_median(scores_list,dist_list)))
          
accuracy = cohen_kappa_score(Y, prediction_list,weights='quadratic') 
print('The accuracy of Using weighted median', accuracy)

