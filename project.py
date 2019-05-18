# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

# Importing the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:,[11,12]].values
y = dataset.iloc[:,88:89].values
X
#Details of dataset
dataset.head()
dataset.isnull().sum()
dataset.dtypes
dataset.describe()
dataset.columns
correlations=dataset.corr(method="pearson")
dataset.skew()
dataset[''].value_counts()

#Labeling categorical datas
dataset=pd.get_dummies(dataset,columns=['Nationality','Club'],prefix=['Nationality','Club'])

#Missing values
dataset['Jersey Number'].fillna(dataset['Jersey Number'].mean(),inplace=True)
dataset['Vision'].fillna(0, inplace=True)
dataset['Vision'].isna().sum()
    
#Editing column contents and datatype
dataset['Value'].replace(regex=True,inplace=True,to_replace=['€','M','K'],value='')
pd.to_numeric(dataset['Value'])
pd.to_datetime(dataset['Joined'])


#Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

skbest=SelectKBest(score_func=chi2, k=4)
fit=skbest.fit(X,y)
fit.scores_
features=fit.transform(X)

___or___

from sklearn.feature_selection import RFE
model=LogisticRegression()
rfe=RFE(model,5)
rfe=rfe.fit(X,y)
rfe.n_features_
rfe.support_

#Feature Importance
model=RandomForestClassifier()
model.fit(X,y)
model.feature_importances_

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test =sc.transform(X_test)

#Importing models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Evaluate models by cross validation
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

#Comparing Algorithms
# Fitting model to the Training set and predicting
from sklearn.metrics import confusion_matrix

def fitmodel(model):
    model.fit(X_train,y_train)
    global y_pred
    y_pred=model.predict(X_test)
    global cm 
    cm= confusion_matrix(y_test, y_pred)
    print(cm)
    
fitmodel(LogisticRegression())

#Classification Report
from sklearn import metrics
from sklearn.metrics import classification_report

metrics.accuracy_score(y_test, y_pred)
classification_report(y_test, y_pred)

# Visualising the Training set results
