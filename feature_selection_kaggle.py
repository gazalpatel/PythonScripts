#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:00:25 2018

@author: gazal
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn import preprocessing

data_path = "/home/gazal/Documents/LapsationRFM/Data/fab_india_customers.csv"
df = pd.read_csv(data_path)
#['id', 'amount_spent', 'avg_latency', 'gender', 'n_products',
#       'n_redemption', 'n_stores', 'n_transactions', 'ATV']
df = df._get_numeric_data()
#['id', 'amount_spent', 'avg_latency', 'n_products', 'n_redemption',
#       'n_stores', 'n_transactions', 'ATV']

#df = df.groupby(['id']).apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

colnames = df.columns.values

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df)
df = pd.DataFrame(np_scaled)
df.columns = colnames
print(df.head(3))

"""
plt.scatter(df.amount_spent, df.avg_latency,
            color='blue', label="amount_spent")
plt.scatter(df.n_products, df.avg_latency,
            color='green', label='n_products')
plt.scatter(df.n_redemption, df.avg_latency,
            color='red', label='n_redemption')

plt.legend(loc="lower right")
plt.title("Sales vs. Advertising")
plt.xlabel("Advertising [1000 $]")
plt.ylabel("Sales [Thousands of units]")
plt.grid()
plt.show()

print(df.corr())
plt.imshow(df.corr(), cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range(len(df.columns))]
plt.xticks(tick_marks, df.columns, rotation='vertical')
plt.yticks(tick_marks, df.columns)

"""
size = df.shape

# Is there a relationship between avg_latency and parameters?

features = df.drop(["id","avg_latency","ATV"], axis=1).columns.values
feature_str = "+".join(features)
formulas = "avg_latency ~" + feature_str 

modelAll = sm.ols(formulas, df).fit()
print(modelAll.params)

# Is at least one of the features useful in predicting Sales?
# H0: There is no relationship between the media and sales versus the alternative hypothesis Ha: There is some relationship between the media and sales.
y_pred = modelAll.predict(df)

RSS = np.sum((y_pred - df.avg_latency)**2)
print("RSS is: ",RSS)
y_mean = np.mean(df.avg_latency) # mean of sales

#Total Sum of Squares (TSS): the total variance in the response Y, and can be thought of as the amount of variability inherent in the response before the regression is performed.
TSS = np.sum((df.avg_latency - y_mean)**2)
print("TSS is : ",TSS)

p=5 # we have three predictors: TV, Radio and Newspaper
n=size[0] # we have 200 data points (input samples)

#The F-statistic is the ratio between (TSS-RSS)/p and RSS/(n-p-1):
# F is far larger than 1: at least one of the three advertising media must be related to sales.
F = ((TSS-RSS)/p) / (RSS/(n-p-1))
print("F is : ",F)

# The quality of a linear regression fit is typically assessed using two related quantities: the residual standard error (RSE) and the R-squared statistic (the square of the correlation of the response and the variable, when close to 1 means high correlation).
RSE = np.sqrt((1/(n-2))*RSS); 
print("RSE is : ",RSE)

R2 = 1 - RSS/TSS; 
print("R square is : ",R2)

print(modelAll.summary())

df = df

def evaluateModel (model):
    print("RSS = ", ((df.avg_latency - model.predict())**2).sum())
    print("R2 = ", model.rsquared)

#features = df.drop(["id","avg_latency","ATV"], axis=1)
#feat_shape = features.shape[1]

for i in range(0,size[1]):
    col = df.iloc[:, i]
    modelTV = sm.ols('avg_latency ~ '+col.name, df).fit()
    print("   ",col.name,"   " )
    #print(modelTV.summary().tables[1])
    print(evaluateModel(modelTV))
    
    
    
    """
        amount_spent    
RSS =  11549.280579316255
R2 =  7.47219542512e-05
None
    n_products    
RSS =  6618.9630750922
R2 =  0.426936730147
None
    n_redemption    
RSS =  11543.798963759747
R2 =  0.000549314802126
None
    n_stores    
RSS =  10164.262673144614
R2 =  0.119988201016
None
    n_transactions    
RSS =  11297.10274840493
R2 =  0.0219080288831
None
    ATV    
RSS =  11486.226066183981
R2 =  0.00553391927354
None

    """