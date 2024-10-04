import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

file_path = os.path.abspath(os.path.join("data", "penguins_pipelined.csv"))
df = pd.read_csv(file_path)
# Assign X, y variables (X is already preprocessed and y is already encoded)
X, y = df.drop('species', axis=1), df['species']
# Create the param_grid and initialize a model
param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 12, 15, 17, 20, 25],
                         'weights': ['distance', 'uniform'],
                         'p': [1, 2, 3, 4, 5]
}
model = KNeighborsClassifier()
# Initialize RandomizedSearchCV and GridSearchCV
randomized = RandomizedSearchCV(model, param_grid, n_iter=100)
grid = GridSearchCV(KNeighborsClassifier(), param_grid)
# Train the GridSearchCV object. During training it finds the best parameters
grid.fit(X, y)
randomized.fit(X, y)
# Print the best estimator and its cross-validation score
print('GridSearchCV:')
print(f"Grid search best estimator: {grid.best_estimator_}")
print(f"Grid search best score: {grid.best_score_}")
print('RandomizedSearchCV:')
print(f"Randomized search best estimator: {randomized.best_estimator_}")
print(f"Grid search best score: {randomized.best_score_}")