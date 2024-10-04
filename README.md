# Machine Learning 
Machine learning workflow.
1. Asking the right questions
2. Preparing data
3. Selecting the algorithm
4. Training the model - Letting specific data teach a Machine Learning algorithm to create a specific prediction model.
5. Testing the modely
   
```
pip install pandas matplotlib scikit-learn
```
Predictable solution based on models built of data
1. Supervised and Unsupervised

In supervised ML, the model contain features and values to be predicted. 

Example: House price predictions
Value: Price
Features: Size:1610, Bedrooms:3, Year Built:2007
Here a model is created using the features and trained against the price

In Unsupervised, cluster of like data is being analyzed to identify groups of data that share the same trait.


# Problem

Predict if a person will get diabetes

# Solution
1. Understand the features in data
2. Identify critical features
3. Focus on at risk population
4. Select data source. Pima India Diabetes study is a good source

# Results
1. Binary result(True or False)
2. Genetic difference are a factor
3. 70% Accuracy is a common target

Using Pima India Diabetes data, predict with 70% or great accuracy, which people will develop diabetes

Disease prediction, Medical research practices, Unknown variations between people, likehood is used

Data Rule #3
Accurately predicting rare events is difficult

Data Rule #4
Track how you manipulate data

# Selecting Algorithm
The algorithm use the result of analytic to create a model that can be trained for future predictions
1. Compare factors - 
   Learning type(prediction model), 
   result(continuous values, price=A*#bedroom+B*size| classification: discrete values, small, medium, large), complexity, basics vs enhanced
   Solution requires predictions. The outcome is binary(True/False)
   Hence, algorithm must support classification(binary classification). This reduces algorithm from 50 < 28 < 20
2. Complexity(Simple. Eliminate "esemble" algorithms, container, multiple child algorithms)
   Boost performance algorithms
   reduces algorithm from 20 to 14. 50 < 28 < 20 < 14
3. Basic vs Enhanced
   Enhanced(variation of basic,performance improvements, additional functionality, more complex)
   Basic(simpler)
   reduces algorithm to 3: 50 < 28 < 20 < 14 < 3[Naive Bayes, Logistic Regression, Decision Tree]

# Algorithms
1. Naive Bayes - Based on likelihood and probability
   Fast - up to 100X faster
   Stable to data changes
2. Logistic Regression - Confusing name, binary result, weights relationship between features resulting in 1/0 output
3. Decision Tree - uses a binary tree structure to make decision. 
   Node contains decision. 
   Requires enough data to determine nodes and splits

Selection is based on Learning(Supervised), Result(Binary classification), Non-ensemble, Basic and finally select Naive Bayes for training

# Training the Model
Datasets are always divided into 2. First 70% for training and the other 30% for testing. 

# Scikit-learn
Designed to work with Numpy, SciPy and Pandas. It contains toolset for training and evaluation tasks:
- Data splitting
- Pre-processing
- Feature Selection
- Model training
- Model tuning







