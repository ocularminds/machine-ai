import joblib
# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

rf_model = RandomForestClassifier(random_state=42, n_estimators=10)      # Create random forest object
rf_model.fit(x_train, y_train.values.flatten())

# Predict Training Data
rf_predict_train = rf_model.predict(x_train)
# training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train, rf_predict_train)))

#Predict Test Data
rf_predict_test = rf_model.predict(x_test)


# training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))
print(metrics.confusion_matrix(y_test, lr_predict_test) )
print("")
print("Classification Report")
print(metrics.classification_report(y_test, lr_predict_test))



# Using the trained Model save trained model to file
# Load trained model from file and Test Prediction on data
joblib.dump(lr_cv_model, "../data/pima-trained-model.pkl")
lr_cv_model = joblib.load("../data/pima-trained-model.pkl")

#Once the model is loaded we can use it to predict on some data. In this case the data file contains a few rows from the original Pima CSV file.
# get data from truncated pima data file
df_predict = pd.read_csv("../data/pima-data-trunc.csv")
print(df_predict.shape)

# df_predict -  The truncated file contained 4 rows from the original CSV.
# Data is the same is in same format as the original CSV file's data. Therefore, just like the original data, we need to transform it before we can make predictions on the data.
# Note: If the data had been previously "cleaned up" this would not be necessary.
# We do this by executed the same transformations as we did to the original data
# Start by dropping the "skin" which is the same as thickness, with different units.

del df_predict['skin']

# df_predict
# We need to drop the diabetes column since that is what we are predicting.
#Store data without the column with the prefix X as we did with the x_train and x_test to indicate that it contains only the columns we are prediction.

X_predict = df_predict
del X_predict['diabetes']
# Data has 0 in places it should not.

# Just like test or test datasets we will use imputation to fix this.
#Impute with mean all 0 readings
fill_0 = SimpleImputer(missing_values=0, strategy="mean")
pd.options.mode.chained_assignment = None
X_predict_cols_not_num_preg = X_predict.columns.difference(['num_preg']) # do not impute num_preg column
X_predict[X_predict_cols_not_num_preg] = fill_0.fit_transform(X_predict[X_predict_cols_not_num_preg])

#X_predict - At this point our data is ready to be used for prediction.
# Predict diabetes with the prediction data. Returns 1 if True, 0 if false
lr_cv_model.predict(X_predict)