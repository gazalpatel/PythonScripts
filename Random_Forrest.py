#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:35:35 2018

@author: gazal
"""

# Pandas is used for data manipulation
import pandas as pd

# Use numpy to convert to arrays
import numpy as np

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot


data_path = "/home/gazal/Documents/LapsationRFM/Data/fab_india_customers.csv"
df = pd.read_csv(data_path)

# Read in data and display first 5 rows
features = df.drop(["id","avg_latency","ATV"], axis=1)
print(features.head(5))

print('The shape of our features is:', features.shape)

# Descriptive statistics for each column
print(features.describe())
# One-hot encode the data using pandas get_dummies
dummies = pd.get_dummies(features)

features_f = pd.concat([features, dummies], axis=1)

# Display the first 5 rows of the last 12 columns
print(features_f.head(5))


# Labels are the values we want to predict
labels = np.array(df['avg_latency'])

# Remove the labels from the features
# axis 1 refers to the columns
features= features_f.drop('gender', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


"""
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# Pull out one tree from the forest
tree = rf.estimators_[5]
"""
"""
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot

# Pull out one tree from the forest
tree = rf.estimators_[5]

# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)

# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')

# Write graph to a png file
graph.write_png('/home/gazal/Documents/LapsationRFM/Data/tree.png')
"""