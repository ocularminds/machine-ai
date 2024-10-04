import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

df = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/a65bbc96-309e-4df9-a790-a1eb8c815a1c/penguins_imputed.csv')
# Assign X, y variables
y = df['species']
X = df.drop('species', axis=1)
# Initialize an ...Encoder object
feature_enc = OneHotEncoder()
# Encode the 'island' and 'sex' columns and add encodings to X
encoded = feature_enc.fit_transform(X[['island', 'sex']]).toarray()
X[['island_Biscoe', 'island_Dream', 'island_Torgersen', 'sex_FEMALE', 'sex_MALE']] = encoded
X.drop(['island', 'sex'], axis=1, inplace=True) # Drop initial 'sex', 'island' columns
# Encode the y
label_enc = LabelEncoder()
y = label_enc.fit_transform(y)
# Print the X
print(X)