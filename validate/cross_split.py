# Build and evaluate a model using both train-test evaluation and cross-validation.
# The data is an already preprocessed Penguins dataset.

import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score

file_path = os.path.abspath(os.path.join("data", "penguins_pipelined.csv"))
df = pd.read_csv(file_path)
# Assign X, y variables (X is already preprocessed and y is already encoded)
X, y = df.drop('species', axis=1), df['species']
# Initialize a model
model = KNeighborsClassifier(n_neighbors=5).fit(X, y)
# Calculate and print the mean of cross validation scores
scores = cross_val_score(model, X, y, cv=3)
print('Cross-val score:', scores.mean())
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# Train a model
model.fit(X_train, y_train)
# Print the score using the test set
print('Train-test score:', model.score(X_test, y_test))