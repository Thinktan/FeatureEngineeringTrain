# import packages we need for exploratory data analysis (EDA)
import pandas as pd  # to store tabular data
import numpy as np  # to do some math
import matplotlib.pyplot as plt  # a popular data visualization tool
import seaborn as sns  # another popular data visualization tool
plt.style.use('fivethirtyeight')  # a popular data visualization theme

# EDA: exploratory data analysis
pima_column_names = ['times_pregnant', 'plasma_glucose_concentration',
                     'diastolic_blood_pressure', 'triceps_thickness',
                     'serum_insulin', 'bmi', 'pedigree_function',
                     'age', 'onset_diabetes']

pima = pd.read_csv('../data/pima.data', names=pima_column_names)
# print(pima.head())

print(pima['onset_diabetes'].value_counts(normalize=True))

# plasma_glucose_concentration 血浆葡萄糖浓度
# bmi 身体质量指数
# diastolic_blood_pressure 舒张压
# for col in ['bmi', 'diastolic_blood_pressure', 'plasma_glucose_concentration']:
#     plt.hist(pima[pima['onset_diabetes']==0][col], 10, alpha=0.5, label='non-diabetes')
#     plt.hist(pima[pima['onset_diabetes']==1][col], 10, alpha=0.5, label='diabetes')
#     plt.legend(loc='upper right')
#     plt.xlabel(col)
#     plt.ylabel('Frequency')
#     plt.title('Histogram of {}'.format(col))
#     plt.show()

#  相关系数矩阵
# sns.heatmap(pima.corr())
# plt.show()

print(pima.corr()['onset_diabetes'])
print(pima.shape)














