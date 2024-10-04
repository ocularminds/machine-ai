# GridSearch and model_selection can be used to improve the performance of a model
# This is done by finding the best hyperparameters fitting our task.
# This process is called Hyperparameter Tuning. 
# The default approach is to try different hyperparameter values and 
# calculate a cross-validation score for them. 
# Then we just choose the value that results in the best score.
# Once you trained a GridSearchCV object, you can use it to make predictions 
# using the .predict() method

# Since GridSearchCV uses all combinations, it is not ideal for large dataset
# RandomizedSearchCV can be used for large dataset


import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

file_path = os.path.abspath(os.path.join("data", "penguins_pipelined.csv"))
df = pd.read_csv(file_path)
# Assign X, y variables (X is already preprocessed and y is already encoded)
X, y = df.drop('species', axis=1), df['species']
# Create the param_grid and initialize GridSearchCV object
param_grid = {'n_neighbors': [1,3,5,7,9]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid)
# Train the GridSearchCV object. During training it finds the best parameters
grid_search.fit(X, y)
# Print the best estimator and its cross-validation score
print(grid_search.best_estimator_)
print(grid_search.best_score_)
print(grid_search.score(X, y))