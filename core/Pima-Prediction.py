# from sklearn.cross_validation import train_test_split deprecated
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import metrics
import pandas as pd  # panda is a dataframe library
import matplotlib.pyplot as plt  # matplotlib plots data
import numpy as np  # numpy provides N-dis object support

# do plotting inline instead of separate window
# %matplotlib inline

# Load and review data
df = pd.read_csv("data\pima-data.csv")
#df = pd.read_csv("data\Pima-Prediction.csv")
df.shape
print(f"data shape: {str(df.shape)}")
print(str(df.head(5)))  # list first 5 rows
print(str(df.tail(5)))  # list last 5 rows

# Eliminate columns not useful, duplicate or with zero values
empty = df.isnull().values.any()
print(f"null or empty columns: {str(empty)}")


def plot_corr(df, size=11):
    """
    Function plots a graphical correction matrix for each pair of columns in the dataframe.

    Input: 
        df: pandas dataframe
        size: vertical and horizontal size of the plot

    Displays: 
        matrix of correction between columns. Blue-cyan-red-darkred => less to more correlated
                                              0...................> 1
                                              Except a darkred line running from top left to bottom right
    """
    corr = df.corr()  # dataframe correction function
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)  # color code the rectangle by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks
    print(corr)

plot_corr(df)
plt.show()
# since the skin column gives the same information as thickness column, it can be removed
del df['skin']
c = plot_corr(df)
print(c)

# For computation let us map the diabates column of True or False to 1 or 0
print("mapping the diabates column of True or False to 1 or 0...")
diabates_map = {True: 1, False: 0}
df['diabetes'] = df['diabetes'].map(diabates_map)
plot_corr(df)
plt.show()
print(df.head(5))

# Check true/false ratio
num_true = len(df.loc[df['diabetes'] == True])
num_false = len(df.loc[df['diabetes'] == False])
percent_true = (num_true / (num_false + num_true)) * 100
percent_false = (num_false / (num_false + num_true)) * 100
print(f"Number of True cases: {percent_true}")
print(f"Number of False cases: {percent_false}")

# Splitting data - 70% for training, 30% for testing
feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp',
                     'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']
x = df[feature_col_names].values  # predictor feature columns (8 x m)
# predicted class (1=True,0=False) column (l x m)
y = df[predicted_class_names].values
split_test_size = 0.30
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=split_test_size, random_state=42
)
percent_training_set = (len(x_train)/len(df.index))*100
percent_testing_set = (len(x_test)/len(df.index))*100
print("{0:0.2f}% in training set".format(percent_training_set))
print("{0:0.2f}% in testing set".format(percent_testing_set))

# verify predicted was split correctly
diabetes_true = len(df.loc[df['diabetes'] == 1])
diabetes_none = len(df.loc[df['diabetes'] == 0])
diabetes_true_trained = len(y_train[y_train[:] == 1])
diabetes_none_trained = len(y_train[y_train[:] == 0])
print("Original True  : {0} ({1:0.2f}%)".format(
    diabetes_true, (diabetes_true/len(df.index)) * 100.0))
print("Original False : {0} ({1:0.2f}%)".format(
    diabetes_none, (diabetes_none/len(df.index)) * 100.0))
print("")
print("Training True  : {0} ({1:0.2f}%)".format(
    diabetes_true_trained, (diabetes_true_trained/len(y_train) * 100.0)))
print("Training False : {0} ({1:0.2f}%)".format(
    diabetes_none_trained, (diabetes_none_trained/len(y_train) * 100.0)))
print("")
print("Test True      : {0} ({1:0.2f}%)".format(
    len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test) * 100.0)))
print("Test False     : {0} ({1:0.2f}%)".format(
    len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test) * 100.0)))

# Post-split data preparation
print(df.head())
# Are these 0 values possible?
# How many rows have have unexpected 0 values?

print("# rows in dataframe {0}".format(len(df)))
print("# rows missing glucose_conc: {0}".format(
    len(df.loc[df['glucose_conc'] == 0])))
print("# rows missing diastolic_bp: {0}".format(
    len(df.loc[df['diastolic_bp'] == 0])))
print("# rows missing thickness: {0}".format(
    len(df.loc[df['thickness'] == 0])))
print("# rows missing insulin: {0}".format(len(df.loc[df['insulin'] == 0])))
print("# rows missing bmi: {0}".format(len(df.loc[df['bmi'] == 0])))
print("# rows missing diab_pred: {0}".format(
    len(df.loc[df['diab_pred'] == 0])))
print("# rows missing age: {0}".format(len(df.loc[df['age'] == 0])))

# Either replace the missing values or ignore them otherwise delete them
# Replace with mean, media or derived value
# NEED CALLOUT MENTION CHANGE TO SIMPLEIMPUTER

# Impute with mean all 0 readings
fill_0 = SimpleImputer(missing_values=0, strategy="mean")
# Notice the missing_values=0 will be replaced by mean.  However, the num_preg can have a value of 0.
# To prevent replacing the 0 num_preg with the mean we need to skip imputing the 'num_preg' column
cols_not_num_preg = pd.DataFrame(x_train).columns.difference(
    ['num_preg'])  # all columns but the num_preg column
# Supress warning message on transformed assignment
pd.options.mode.chained_assignment = None

# impute the training data
x_train[cols_not_num_preg] = fill_0.fit_transform(x_train[cols_not_num_preg])

# impute the test data
x_test[cols_not_num_preg] = fill_0.transform(x_test[cols_not_num_preg])

# Training Initial Algorithm - Naive Bayes

# create Gaussian Naive Bayes model object and train it with the data
nb_model = GaussianNB()
nb_model.fit(x_train, pd.DataFrame(y_train).values.flatten())

# Performance on Training Data
# predict values using the training data
nb_predict_train = nb_model.predict(x_train)

# Accuracy
print("Accuracy with training data: {0:.4f}".format(
    metrics.accuracy_score(y_train, nb_predict_train)))
print()

# Performance on Testing Data
# predict values using the testing data
nb_predict_test = nb_model.predict(x_test)

# training metrics
#print("nb_predict_test", nb_predict_test)
#print ("y_test", y_test)
print("Accuracy with test data: {0:.4f}".format(
    metrics.accuracy_score(y_test, nb_predict_test)))

# Metrics
print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test, nb_predict_test)))
print("")

print("Classification Report - Naive Bayes")
print(metrics.classification_report(y_test, nb_predict_test))

# Using Logistic Regression algorithm

lr_model = LogisticRegression(
    C=0.7, random_state=42, solver='liblinear', max_iter=10000
)
lr_model.fit(x_train, pd.DataFrame(y_train).values.flatten())  # .ravel())
lr_predict_test = lr_model.predict(x_test)

# training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))
print(metrics.confusion_matrix(y_test, lr_predict_test))
print("")
print("Classification Report - Logistic Regression")
print(metrics.classification_report(y_test, lr_predict_test))


#Setting regularization parameter
C_start = 0.1
C_end = 5
C_inc = 0.1

C_values, recall_scores = [], []

C_val = C_start
best_recall_score = 0
while (C_val < C_end):
    C_values.append(C_val)
    lr_model_loop = LogisticRegression(C=C_val, random_state=42, solver='liblinear')
    lr_model_loop.fit(x_train, pd.DataFrame(y_train).values.flatten()) #.ravel())
    lr_predict_loop_test = lr_model_loop.predict(x_test)
    recall_score = metrics.recall_score(y_test, lr_predict_loop_test)
    recall_scores.append(recall_score)
    if (recall_score > best_recall_score):
        best_recall_score = recall_score
        best_lr_predict_test = lr_predict_loop_test
        
    C_val = C_val + C_inc

best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print("1st max value of {0:.3f} occured at C={1:.3f}".format(best_recall_score, best_score_C_val))

#%matplotlib inline 
plt.plot(C_values, recall_scores, "-")
plt.xlabel("C value")
plt.ylabel("recall score")
plt.show()

# Logisitic regression with class_weight='balanced'
C_start = 0.1
C_end = 5
C_inc = 0.1

C_values, recall_scores = [], []

C_val = C_start
best_recall_score = 0
while (C_val < C_end):
    C_values.append(C_val)
    lr_model_loop = LogisticRegression(C=C_val, class_weight="balanced", random_state=42, solver='liblinear', max_iter=10000)
    lr_model_loop.fit(x_train, pd.DataFrame(y_train).values.flatten()) 
    lr_predict_loop_test = lr_model_loop.predict(x_test)
    recall_score = metrics.recall_score(y_test, lr_predict_loop_test)
    recall_scores.append(recall_score)
    if (recall_score > best_recall_score):
        best_recall_score = recall_score
        best_lr_predict_test = lr_predict_loop_test
        
    C_val = C_val + C_inc

best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print("1st max value of {0:.3f} occured at C={1:.3f}".format(best_recall_score, best_score_C_val))

#%matplotlib inline 
plt.plot(C_values, recall_scores, "-")
plt.xlabel("C value")
plt.ylabel("recall score")
plt.show()
from sklearn.linear_model import LogisticRegression
lr_model =LogisticRegression( class_weight="balanced", C=best_score_C_val, random_state=42, solver='liblinear')
lr_model.fit(x_train, pd.DataFrame(y_train).values.flatten())
lr_predict_test = lr_model.predict(x_test)

# training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))
print(metrics.confusion_matrix(y_test, lr_predict_test) )
print("")
print("Classification Report")
print(metrics.classification_report(y_test, lr_predict_test))
print(metrics.recall_score(y_test, lr_predict_test))

#K-Fold Cross Validation allow splitting dtata to three: train, test, validation.
#here the test data is split to 10 and 1/10 of the data is used to validate the model
#LogisticRegressionCV
from sklearn.linear_model import LogisticRegressionCV
lr_cv_model = LogisticRegressionCV(n_jobs=-1, random_state=42, Cs=3, cv=10, refit=False, class_weight="balanced", max_iter=500)  # set number of jobs to -1 which uses all cores to parallelize
lr_cv_model.fit(x_train, pd.DataFrame(y_train).values.flatten())
lr_cv_predict_test = lr_cv_model.predict(x_test)

# training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, lr_cv_predict_test)))
print(metrics.confusion_matrix(y_test, lr_cv_predict_test) )
print("")
print("Classification Report")
print(metrics.classification_report(y_test, lr_cv_predict_test))
