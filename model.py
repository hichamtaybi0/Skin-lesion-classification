# import the necessary packages
import h5py
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import pickle

import warnings
warnings.filterwarnings('ignore')



# import the feature vector and trained labels
x_train_data = h5py.File('Extracted_Features/x_train.h5', 'r')
y_train_data = h5py.File('Extracted_Features/y_train.h5', 'r')
x_test_data = h5py.File('Extracted_Features/x_test.h5', 'r')
y_test_data = h5py.File('Extracted_Features/y_test.h5', 'r')

x_train = x_train_data['dataset_1']
y_train = y_train_data['dataset_1']
x_test = x_train_data['dataset_1']
y_test = y_train_data['dataset_1']

X_train = np.array(x_train)
y_train = np.array(y_train)
X_test = np.array(x_test)
y_test = np.array(y_test)

'''
# ********************************
# concat data if cross validation used
# ********************************
X_train = np.concatenate((X_train, X_test), axis=0)
y_train = np.concatenate((y_train, y_test), axis=0)
'''

x_train_data.close()
y_train_data.close()
x_test_data.close()
y_test_data.close()


# variables to hold the results and names
results = []
names = []
scoring = "accuracy"


model = DecisionTreeClassifier(random_state=9)
#model = LogisticRegression()

# 10-fold cross validation
#kfold = KFold(n_splits=100, random_state=7)
#cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
#results.append(cv_results)
#msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
#print(msg)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)

print(conf_mat)

# save the model to disk
pickle.dump(model, open('finalized_model.sav', 'wb'))
