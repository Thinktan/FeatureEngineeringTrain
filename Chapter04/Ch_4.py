
import pandas as pd

X = pd.DataFrame({'city':['tokyo', None, 'london', 'seattle', 'san francisco', 'tokyo'],
                  'boolean':['yes', 'no', None, 'no', 'no', 'yes'],
                  'ordinal_column':['somewhat like', 'like', 'somewhat like', 'like', 'somewhat like', 'dislike'],
                  'quantitative_column':[1, 11, -.5, 10, None, 20]})

print(X)