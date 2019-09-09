#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 12:42:31 2018

@author: gazal
"""
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

data_path = "/home/gazal/Documents/LapsationRFM/Data/fab_india_customers.csv"
df = pd.read_csv(data_path)

#df = df._get_numeric_data() #drop non-numeric cols
print(df.head())

# Put the target (housing value -- MEDV) in another DataFrame

features = df.drop(["id","avg_latency","ATV"], axis=1).columns.values
feature_str = "+".join(features)

#X = df[feature_str]
#print(X.head(3))
#y = target["MEDV"]
#target = df["avg_latency"]
formulas = "avg_latency ~" + feature_str 

# Note the difference in argument order
#model = sm.OLS(formulas).fit()
model = ols(formulas, df).fit()
print(model.summary())


#predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()
