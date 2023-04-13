from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import pandas as pd
# import the sentence tokenizer from nltk
from nltk.tokenize import sent_tokenize
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from time import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

hotel_reviews = pd.read_csv('../data/7282_1.csv')

print(hotel_reviews.shape)

# print(hotel_reviews.head())

# hotel_reviews.plot.scatter(x='longitude', y='latitude')
# plt.show()

#Filter to only include datapoints within the US
hotel_reviews = hotel_reviews[((hotel_reviews['latitude']<=50.0) \
                               & (hotel_reviews['latitude']>=24.0)) \
                              & ((hotel_reviews['longitude']<=-65.0) \
                                 & (hotel_reviews['longitude']>=-122.0))]

# Plot the lats and longs again
# hotel_reviews.plot.scatter(x='longitude', y='latitude')
# plt.show()

texts = hotel_reviews['reviews.text']
sentences = reduce(lambda x, y:x+y, texts.apply(lambda x: sent_tokenize(str(x))))
print(len(sentences)) # 118239
print(sentences[0])


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')

tfidf_transformed = tfidf.fit_transform(sentences)

print(tfidf_transformed.shape) # (118239, 280900)


tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
svd = TruncatedSVD(n_components=10)  # will extract 10 "topics"
normalizer = Normalizer() # will give each document a unit norm

lsa = Pipeline(steps=[('tfidf', tfidf), ('svd', svd), ('normalizer', normalizer)])

lsa_sentences = lsa.fit_transform(sentences)

print(lsa_sentences.shape)

cluster = KMeans(n_clusters=10)
cluster.fit(tfidf_transformed)
cluster.predict(tfidf_transformed)

cluster.fit(lsa_sentences)
cluster.predict(lsa_sentences)

predicted_cluster = cluster.predict(lsa_sentences)
# predicted_cluster

# Distribution of "topics"
pd.Series(predicted_cluster).value_counts(normalize=True)# create DataFrame of texts and predicted topics
texts_df = pd.DataFrame({'text':sentences, 'topic':predicted_cluster})

texts_df.head()

print("Top terms per cluster:")
original_space_centroids = svd.inverse_transform(cluster.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = lsa.steps[0][1].get_feature_names()
for i in range(10):
    print("Cluster %d:" % i)
    print(', '.join([terms[ind] for ind in order_centroids[i, :5]]))
    print()

print(lsa.steps[0][1])







