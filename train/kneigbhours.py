import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os;

# Build a KNeighborsClassifier, train it, and get its accuracy using the .score() method.
file_path = os.path.abspath(os.path.join("data", "penguins_pipelined.csv"))
df = pd.read_csv(file_path)
# Assign X, y variables (X is already preprocessed and y is already encoded)
X, y = df.drop('species', axis=1), df['species']

# Initialize and train a model
knn5 = KNeighborsClassifier().fit(X, y) # Trained 5 neighbors model
knn1 = KNeighborsClassifier(n_neighbors=1).fit(X, y) # Trained 1 neighbor model

# Print the scores of both models
print('5 Neighbors score:', knn5.score(X, y))
print('1 Neighbor score:', knn1.score(X, y))

# Split data to train and test and run the score against test data
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# Initialize and train a model
knn5 = KNeighborsClassifier().fit(X_train, y_train) # Trained 5 neighbors model
knn1 = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train) # Trained 1 neighbor model
# Print the scores of both models
print("\nscoring against test data")
print('5 Neighbors score:',knn5.score(X_test, y_test))
print('1 Neighbor score:',knn1.score(X_test, y_test))
