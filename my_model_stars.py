# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 18:47:40 2022

@author: prach
"""

import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, precision_recall_fscore_support
import pickle


df_data = pd.read_csv(r"C:\Users\prach\OneDrive\Desktop\week4-data-glacier\Stars-random.csv")

data = df_data.drop(columns=['Type']).values

out = (df_data['Type']).values

train_data, test_data, train_out, test_out = train_test_split(data, out)

regressor = LogisticRegression(max_iter=1000)

regressor.fit(train_data, train_out)

out_predicted = regressor.predict(test_data)

print('Precision                                   : %.3f'%precision_score(test_out, out_predicted, average = 'micro'))
print('Recall                                      : %.3f'%recall_score(test_out, out_predicted, average = 'micro'))
print('F1-Score                                    : %.3f'%f1_score(test_out, out_predicted, average = 'micro'))
print('\nPrecision Recall F1-Score Support Per Class : \n',precision_recall_fscore_support(test_out, out_predicted, average = 'micro'))
print('\nClassification Report                       : ')
print(classification_report(test_out, out_predicted))

# saving the model 
pickle.dump(regressor, open('model_stars_data.pkl','wb'))

model_cancer = pickle.load(open('model_cancer_data.pkl','rb'))

