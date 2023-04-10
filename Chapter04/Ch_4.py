
import pandas as pd

# X = pd.DataFrame({'city':['tokyo', None, 'london', 'seattle', 'san francisco', 'tokyo'],
#                   'boolean':['yes', 'no', None, 'no', 'no', 'yes'],
#                   'ordinal_column':['somewhat like', 'like', 'somewhat like', 'like', 'somewhat like', 'dislike'],
#                   'quantitative_column':[1, 11, -.5, 10, None, 20]})
# print(X)


tweets = pd.read_csv('../data/twitter_sentiment.csv', encoding='latin1')

# print(tweets.head())
del tweets['ItemID']
print(tweets.head())

X = tweets['SentimentText']
y = tweets['Sentiment']
print(tweets.shape)
print(X.shape)

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
_ = vect.fit_transform(X)
print(_.shape)

from






















