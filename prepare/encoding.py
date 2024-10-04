import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/a65bbc96-309e-4df9-a790-a1eb8c815a1c/adult_edu.csv')

# print unique categories
print(df['education'].unique())

# Create a list of ordered categorical values, from 'HS-grad' to 'Doctorate'.
# Load the data and assign X, y variables
df = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/a65bbc96-309e-4df9-a790-a1eb8c815a1c/adult_edu.csv')
y = df['income'] # 'income' is a target in this dataset
X = df.drop('income', axis=1)
# Create a list of categories so HS-grad is encoded as 0 and Doctorate as 6
edu_categories = ['HS-grad', 'Some-college', 'Assoc', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate']
# Initialize an OrdinalEncoder instance with the correct categories
ord_enc = OrdinalEncoder(categories=[edu_categories])
# Transform the 'education' column and print it
X['education'] = ord_enc.fit_transform(X[['education']])
print(X['education'])

# using HotEncoder
df2 = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/a65bbc96-309e-4df9-a790-a1eb8c815a1c/penguins_imputed.csv')

print('island: ', df2['island'].unique())
print('sex: ', df2['sex'].unique())

df = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/a65bbc96-309e-4df9-a790-a1eb8c815a1c/penguins_imputed.csv')
# Assign X, y variables
y = df['species']
X = df.drop('species', axis=1)
# Initialize an OneHotEncoder object
one_hot = OneHotEncoder()
# Print transformed 'sex', 'island' columns
print(one_hot.fit_transform(X[['sex', 'island']]).toarray())

# Encoding the y axis with label encoder
# The LabelEncoder is used to encode the target, regardless of whether it is nominal or ordinal.

# Load the data and assign X, y variables
df = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/a65bbc96-309e-4df9-a790-a1eb8c815a1c/adult_edu.csv')
y = df['income'] # Income is a target in this dataset
X = df.drop('income', axis=1)

print(y)
print('All values: ', y.unique())

# Load the data and assign X, y variables
df = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/a65bbc96-309e-4df9-a790-a1eb8c815a1c/adult_edu.csv')
y = df['income'] # Income is a target in this dataset
X = df.drop('income', axis=1)
# Initialize a LabelEncoder object and encode the y variable
label_enc = LabelEncoder()
y = label_enc.fit_transform(y)
print(y)
# Decode the y variable back
y_decoded = label_enc.inverse_transform(y)
print(y_decoded)