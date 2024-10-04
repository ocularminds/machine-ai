import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/a65bbc96-309e-4df9-a790-a1eb8c815a1c/penguins.csv')
# Removing rows with more than 1 null
df = df[df.isna().sum(axis=1) < 2] 
# Assigning X, y variables
X, y = df.drop('species', axis=1), df['species']
# Encode the target
label_enc = LabelEncoder()
y = label_enc.fit_transform(y)
# Create the ColumnTransformer for encoding features
ct = make_column_transformer((OneHotEncoder(), ['island', 'sex']), remainder='passthrough')
# Make a Pipeline of ct, SimpleImputer, and StandardScaler
pipe = make_pipeline(ct, SimpleImputer(strategy='most_frequent'),
					 StandardScaler(), KNeighborsClassifier())
# Train the model
pipe.fit(X, y)
# Print predictions
y_pred = pipe.predict(X) # Get encoded predictions
print(label_enc.inverse_transform(y_pred)) # Decode predictions and print