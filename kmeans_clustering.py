#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 11:13:21 2018

@author: gazal
"""


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import pylab as pl
from sklearn.decomposition import PCA

# Importing the dataset
data_path = "/home/gazal/Documents/LapsationRFM/Data/fab_india_customers.csv"
df = pd.read_csv(data_path)
features = df.drop(["id","avg_latency","ATV"], axis=1)
X = pd.get_dummies(features)
y = df[['avg_latency']]

"""
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
"""

Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
print(kmeans)
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
print(score)

pl.plot(Nc,score)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()

pca = PCA(n_components=1).fit(X)
pca_d = pca.transform(X)
pca_c = pca.transform(y)

kmeans=KMeans(n_clusters=3)
kmeansoutput=kmeans.fit(y)
print(kmeansoutput)

pl.figure('3 Cluster K-Means')
pl.scatter(pca_c[:, 0], pca_d[:, 0], c=kmeansoutput.labels_)
pl.xlabel('Dividend Yield')
pl.ylabel('Returns')
pl.title('3 Cluster K-Means')
pl.show()