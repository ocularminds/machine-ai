import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/a65bbc96-309e-4df9-a790-a1eb8c815a1c/penguins.csv')
# Removing rows with more than 1 null
df = df[df.isna().sum(axis=1) < 2] 
# Transform the 'sex' column, mean for numeric, most for category
imputer = SimpleImputer(strategy='most_frequent') 
df['sex'] = imputer.fit_transform(df[['sex']]).ravel()
print(df.head(8))