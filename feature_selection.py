#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 13:41:24 2018

@author: gazal
"""

from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
from statsmodels.formula.api import ols

data_path = "/home/gazal/Documents/LapsationRFM/Data/fab_india_customers.csv"
df = pd.read_csv(data_path)
df = df._get_numeric_data()
features = df.drop(["id","avg_latency","ATV"], axis=1).columns.values
feature_str = "+".join(features)

formulas = "avg_latency ~" + feature_str 

#model = ols(formulas, df).fit()

features = df.drop(["id","avg_latency","ATV"], axis=1)
label = df["avg_latency"]

# create the RFE model and select 3 attributes
#rfe = RFE(model, 3)
#rfe = rfe.fit(features, label)

model = ExtraTreesClassifier()
model.fit(features, label)
# display the relative importance of each attribute
print(model.feature_importances_)