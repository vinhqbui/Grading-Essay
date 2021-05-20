# -*- coding: utf-8 -*-
"""
Created on Mon May 17 20:20:22 2021

@author: buiqu
"""
import pandas as pd
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

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
train = pd.read_excel('./Data/training_set_rel3_set1.xlsx')
test = pd.read_excel('./Data/valid_set_set1.xlsx')
y_train = train['Score']
y_test = test['Score']
X = pd.concat([train,test], ignore_index = True)

train_numberOfSentences = X['Essay Content'].apply(lambda x: len(x.split('.')))
train_numberOfWords = X['Essay Content'].apply(lambda x: len(x.split()))
content = X['Essay Content']
content = content.apply(lambda x: re.sub('[^a-zA-Z]+', ' ', x))
content = content.apply(lambda x: x.lower())

content = content.apply(lambda x: WordNetLemma(word_tokenize(x)))
content = content.apply(lambda x: RemoveStopWords(word_tokenize(x)))
#content = content.apply(lambda x: SnowballStemmer(word_tokenize(x)))

#sns.set_theme()
heatmapData = list()
for dimensions in range(80, 351, 10):
    accuracyOfDimension = list()
    svd = TruncatedSVD(n_components=dimensions)
    tfidf = TfidfVectorizer(min_df = 0.01, max_df=0.85, stop_words='english')
    
    x_transform = tfidf.fit_transform(content)
    x_transform = sparse.hstack((x_transform, train_numberOfSentences[:,None]))
    x_transform = sparse.hstack((x_transform, train_numberOfWords[:,None]))
    
    #x_transform = SVD_for_S.SVD(x_transform.toarray(), dimensions)
    x_transform = svd.fit_transform(x_transform)
    
    x_train = x_transform[:len(train)]
    x_test = x_transform[len(train):]
    for neighbors in range(3, 11, 1):     
        nearestNeighbors = NearestNeighbors(n_neighbors=neighbors)
        nearestNeighbors.fit(x_train)
        test_dist, test_ind = nearestNeighbors.kneighbors(x_test)
        
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
        accuracyOfDimension.append(accuracy)
    heatmapData.append(accuracyOfDimension)
    
df = pd.DataFrame(heatmapData, columns=list(range(3,11,1)), index=list(range(80, 351, 10)))
plt.figure(figsize=(16, 16))
sns.heatmap(df)