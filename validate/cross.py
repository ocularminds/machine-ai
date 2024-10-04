# Cross-validation is usually used to determine the best hyperparameters 
# (e.g., the best number of neighbors).
# Use cross_validate to split data into 5 folds(default) 
# Chose 1 fold for test and the rest folds to train
# Finally, find the mean score.

import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


file_path = os.path.abspath(os.path.join("data", "penguins_pipelined.csv"))
df = pd.read_csv(file_path)
# Assign X, y variables (X is already preprocessed and y is already encoded)
X, y = df.drop('species', axis=1), df['species']
# Print the cross-val scores and the mean for KNeighborsClassifier with 5 neighbors
scores = cross_val_score(KNeighborsClassifier(), X, y)
print(scores)
print(scores.mean())