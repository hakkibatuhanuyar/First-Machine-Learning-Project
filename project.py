# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

# Importing the dataset
dataset = pd.read_csv('heart.csv')
X = dataset.iloc[:,0:13].values
y = dataset.iloc[:,13:14].values

# Details of dataset
dataset.head()
dataset.describe()
dataset.columns
dataset.plot(kind="box",subplots=True,sharex=False,sharey=False)
plt.show()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test =sc.transform(X_test)

# Importing models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Evaluate models by cross validation
from sklearn.model_selection import KFold,cross_val_score
import operator

models=[[DecisionTreeClassifier(),"DecisionTree"],
        [RandomForestClassifier(),"RandomForest"],
        [LogisticRegression(),"LogisticRegression"],
        [KNeighborsClassifier(),"KNN"],
        [GaussianNB(),"NaiveBayes"],
        [SVC(),"SVM"],
        [XGBClassifier(),"XGBoost"]]
results={}
kfold=KFold(n_splits=10,random_state=10)
for i in range(len(models)):
    result=cross_val_score(models[i][0],X_train,y_train,cv=kfold)
    results[models[i][1]]=str(result.mean())
print("Maksimum skor : {}, {}".format(max(results.items(), key=operator.itemgetter(1))[0],
                                      max(results.items(), key=operator.itemgetter(1))[1]))

# Comparing Algorithms
# Fitting model to the Training set and predicting
from sklearn.metrics import confusion_matrix
def fitmodel(model):
    model.fit(X_train,y_train)
    global y_pred
    y_pred=model.predict(X_test)
    global cm 
    cm= confusion_matrix(y_test, y_pred)
    print(cm)
    
fitmodel(LogisticRegression(random_state=0))

# Visualising the Training set results
