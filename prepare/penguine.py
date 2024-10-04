import pandas as pd

df = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/a65bbc96-309e-4df9-a790-a1eb8c815a1c/penguins.csv')

print(df.head(10))
# check missing values
df.info()
# print missing info
print(df[df.isna().any(axis=1)])

# to clean, remove row with at most 2 Nan
df = df[df.isna().sum(axis=1) < 2]
print(df.head(8))