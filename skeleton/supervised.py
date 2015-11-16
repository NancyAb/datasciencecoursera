import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import KFold, train_test_split
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler

def prep_data(X, y):

    return train_feature, test_feature, train_target, test_target = \
                train_test_split(X, y, test_size=0.3)

def k_fold(n_fo, X, y):
        kf = KFold(len(X), n_folds = n_fo)
        train, test = list(kf)[0]
        each_fold_prec, each_fold_acc, each_fold_rec = [], [], []
        np.random.seed = 123
        np.random.shuffle(np.array(X))
        np.random.shuffle(np.array(y))
        for train, test in kf:
                model = LogisticRegression()
                model = model.fit(np.array(X)[train], np.array(y)[train])
                y_predict = model.predict(np.array(X)[test])
                each_fold_rec += [metrics.recall_score(np.array(y)[test], y_predict)]
                each_fold_acc += [metrics.accuracy_score(np.array(y)[test], y_predict)]
                each_fold_prec += [metrics.precision_score(np.array(y)[test], y_predict)]
        return np.array(each_fold_rec).mean(), np.array(each_fold_acc).mean(), np.array(each_fold_prec).mean()

if __name__ == '__main__':
    data = np.genfromtxt('data/spam.csv', delimiter=',')

    y = data[:, -1]
    x = data[:, 0:-1]

    train_x, test_x, train_y, test_y = train_test_split(x, y)


'''
# 2. Convert the "no", "yes" values to booleans (True/False)
df["Int'l Plan"] = df["Int'l Plan"] == 'yes'

# 3. Remove the features which aren't continuous or boolean
df.pop('State')

# 4. Make a numpy array called y containing the churn values
y = df.pop('Churn?').values

# 7. Use sklearn's RandomForestClassifier to build a model of your data
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# 8. What is the accuracy score on the test data?
print "8. score:", rf.score(X_test, y_test)
## answer: 0.9448441247

# 9. Draw a confusion matrix for the results
y_predict = rf.predict(X_test)
print "9. confusion matrix:"
print confusion_matrix(y_test, y_predict)
## answer:  716   6
##           40  72

# 10. What is the precision? Recall?
print "10. precision:", precision_score(y_test, y_predict)
print "    recall:", recall_score(y_test, y_predict)

# 11. Build the RandomForestClassifier again setting the out of bag parameter to be true
rf = RandomForestClassifier(n_estimators=30, oob_score=True)
rf.fit(X_train, y_train)
print "11: accuracy score:", rf.score(X_test, y_test)
print "    out of bag score:", rf.oob_score_
##   accuracy score: 0.953237410072
## out of bag score: 0.946778711485   (out-of-bag error is slightly worse)

# 12. Use sklearn's model to get the feature importances
feature_importances = np.argsort(rf.feature_importances_)
print "12: top five:", list(df.columns[feature_importances[-1:-6:-1]])
## top five: ['Day Mins', 'CustServ Calls', 'Day Charge', "Int'l Plan", 'Eve Mins']
## (will vary a little)

# 13. Calculate the standard deviation for feature importances across all trees

n = 10 # top 10 features

importances = forest_fit.feature_importances_[:n]
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
             indices = np.argsort(importances)[::-1]

'''
