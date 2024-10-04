import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

df = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/a65bbc96-309e-4df9-a790-a1eb8c815a1c/exams.csv')
# Ordered categories of parental level of education for OrdinalEncoder
edu_categories = ['high school', 'some high school', 'some college', "associate's degree", "bachelor's degree", "master's degree"]
# Making a column transformer
ct = make_column_transformer(
  (OrdinalEncoder(categories=[edu_categories]), ['parental level of education']),
  (OneHotEncoder(), ['gender', 'race/ethnicity', 'lunch', 'test preparation course']), 
  remainder='passthrough'
)

print(ct.fit_transform(df))