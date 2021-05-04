# Grading-Essay
### For more information on this project, please visit my [blog](https://medium.com/@bui.quoc.vinh.2010/nlp-automatic-essay-grading-using-tf-idf-svd-and-k-nearest-neighbors-classifiers-by-vinh-bui-1b2c0b1e1a4a).

## 1. What is the project?
The project's data can be found here: https://www.kaggle.com/c/asap-aes. The original project was hosted 8 years ago. In this project, I try to beat the highest score which at the time was ~81%. Machine learning has come a long way from what it was 8 years ago. As such beating the original score does not mean much, but it is still a great project to practice machine learning skills. There are many people who could achieve more than 95% accuracy for this project using neuron network. However, I will not be using neuron network as it is harder to explain how the model produces the result to stakeholders. Instead I will use feature engineering and K-Nearest Neighbors.
## 2. Stucture of project
 - Folder Data: there is processed data that I use.
 - Using TFIDF - SVD to predict.ipynb: This is Jupyter Notebook for the projects.
## 3. Concepts used in this project
 Additional resources can be found by following the links.
 - TF-IDF: [Term frequency-inverse document frequency](http://www.tfidf.com/)
 - SVD: [Singular Value Decomposion](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
 - [K Nearest Neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html?highlight=nearestneighbor#sklearn.neighbors.NearestNeighbors) 
## 4. Result
This model has an accuracy of 87%, when calculated with the cohen cappa quadratic.
