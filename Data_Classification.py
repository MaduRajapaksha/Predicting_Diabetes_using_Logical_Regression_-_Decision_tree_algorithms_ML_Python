import numpy 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from math import sqrt

# render the plot inline, instead of in a separate window
get_ipython().run_line_magic('matplotlib', 'inline')

# loading the data set
df = pd.read_csv("D:/L4S1/DataMining/MiniProject/data/pima-data.csv")

# take a look at the shape of the data set
df.shape


# finding the correlated columns
def plot_corr(df, size=11):
    corr = df.corr()  # calling the correlation function on the datafrmae
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)  # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tickmarks
    plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tickmarks


df.corr()

# Deleting one of a correlatedcolumn
del df['skin']

# map the predicted value as 1 & 0
diabetes_map = {True: 1, False: 0}
df['diabetes'] = df['diabetes'].map(diabetes_map)

# columns
feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']

# prediction factor
X = df[feature_col_names].values

# this is what we want to predict
y = df[predicted_class_names].values

split_test_size = 0.3

# replace the 0 values into mean of the dataset
df[['glucose_conc', 'diastolic_bp', 'bmi']] = df[['glucose_conc', 'diastolic_bp', 'bmi']].replace(0, numpy.NaN)
df.fillna(df.mean(), inplace=True)

# split the data set into test set and the training set
# 42 is the set.seed() equivalent in Python which generates repeatable random distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42)

# LOGISTIC REGRESSION MODEL
lr_model = LogisticRegression(C=0.7, random_state=42)
lr_model.fit(X_train, y_train.ravel())
lr_predict_test = lr_model.predict(X_test)

# training metrics
print("Accuracy:{0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))
print()
print("Accuracy on training set: {:.4f}".format(lr_model.score(X_train, y_train)))
print("Accuracy on test set: {:.4f}".format(lr_model.score(X_test, y_test)))
print("Confusion Matrix")
print(metrics.confusion_matrix(y_test, lr_predict_test))
print()
print("Classification Report")
print(metrics.classification_report(y_test, lr_predict_test))

# Calculate the mean squared error
rms = sqrt(mean_squared_error(y_test, lr_predict_test))

print("RMSE = ", rms)

# improve rEGRESSION method by changing the regularization parameter for logistic regression model
# This section will try C value from 0.1 to 4.9 in increments of 0.1.
# For each C-value, it will create a logistic regression and train with the train data. 
# Afterwards, it will predict the test data for the different C-values, and the highest result is recorded.

C_start = 0.1
C_end = 5
C_inc = 0.1

C_values, recall_scores = [], []

C_val = C_start
best_recall_score = 0
while (C_val < C_end):
    C_values.append(C_val)
    lr_model_loop = LogisticRegression(C=C_val, random_state=42)
    lr_model_loop.fit(X_train, y_train.ravel())
    lr_predict_loop_test = lr_model_loop.predict(X_test)
    recall_score = metrics.recall_score(y_test, lr_predict_loop_test)
    recall_scores.append(recall_score)
    if (recall_score > best_recall_score):
        best_recall_score = recall_score
        best_lr_predict_test = lr_predict_loop_test

    C_val = C_val + C_inc

best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print("1st max value of {0:.3f} occured at C={1:.3f}".format(best_recall_score, best_score_C_val))

# plot the changes in C-values against recall scores to see how the regularization scores impact the recall score
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(C_values, recall_scores, "-")
plt.xlabel("C value")
plt.ylabel("recall score")

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

# max_depth set to 3
#tree = DecisionTreeClassifier(max_depth=3, random_state=0)
#tree.fit(X_train, y_train)
#print("Accuracy on training set: {:.4f}".format(tree.score(X_train, y_train)))
#print("Accuracy on test set: {:.4f}".format(tree.score(X_test, y_test)))
# taking the feature importance
print("Feature importances:\n{}".format(tree.feature_importances_))

tree.fit(X_train, y_train.ravel())
tree_predict_test = tree.predict(X_test)

# training metrics
print("Accuracy:{0:.4f}".format(metrics.accuracy_score(y_test, tree_predict_test)))
print()
print("Confusion Matrix")
print(metrics.confusion_matrix(y_test, tree_predict_test))
print()
print("Classification Report")
print(metrics.classification_report(y_test, tree_predict_test))


# plot the feature importances
def plot_feature_importances_diabetes(model):
    plt.figure(figsize=(8, 6))
    n_features = 8
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), df)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)


plot_feature_importances_diabetes(tree)
plt.savefig('feature_importance')

# MSE
rms = sqrt(mean_squared_error(y_test, tree_predict_test))
print("RMSE = ", rms)