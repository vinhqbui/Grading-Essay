# Grading-Essay
### For more information, please visit my [Medium blog](https://medium.com/@bui.quoc.vinh.2010/nlp-automatic-essay-grading-using-tf-idf-svd-and-k-nearest-neighbors-classifiers-by-vinh-bui-1b2c0b1e1a4a).

## 1. What is the project?
The project's data can be found here: https://www.kaggle.com/c/asap-aes.
Even though the project is hosted 8 years ago. It is still a great resources for practicing machine learning skills. 
In this project, I try to beat the highest score at the time (~81%). The machine learning right now is much better than 8 years ago, so beating it does not mean anything. There are many people could achieve more than 95% accuracy for this project using neuron network. However, I will not use it because neuron network is hard to explain how the model produces the result to stakeholders, so I will use features engineering and K-Nearest Neighbors.
## 2. Stucture of project
 - Folder Data: there is processed data that I use.
 - Using TFIDF - SVD to predict.ipynb: This is Jupyter Notebook for the projects.
## 3. Concept using in this project
 You can look for reference in the following links.
 - TF-IDF: [Term frequency-inverse document frequency](http://www.tfidf.com/)
 - SVD: [Singular Value Decomposion](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
 - [K Nearest Neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html?highlight=nearestneighbor#sklearn.neighbors.NearestNeighbors) 
## 4. Result
With cohen cappa score quadractic, the model yells accuracy 87%. 
