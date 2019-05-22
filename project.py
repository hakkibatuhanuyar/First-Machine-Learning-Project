# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

# Importing the dataset
dataset = pd.read_csv('credit.csv')
X = dataset.iloc[:,dataset.columns!='class'].values
y = dataset.iloc[:,7:8].values

#Details of dataset
dataset.head()
dataset.isnull().sum()
dataset.dtypes
dataset.describe()
dataset.columns
correlations=dataset.corr(method="pearson")
dataset.skew()
dataset[''].value_counts()
dataset.groupby('').size()
dataset[['','','']].sort_values(by='',ascending=True)

#Labeling categorical datas
from sklearn.preprocessing import LabelEncoder
y=LabelEncoder().fit_transform(y)

dataset=pd.get_dummies(dataset,columns=['checking_status','credit_history','purpose','savings_status','employment','personal_status','other_parties','property_magnitude','other_payment_plans','housing','job','own_telephone','foreign_worker'])

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

features=SelectKBest(score_func=chi2, k=30).fit_transform(X,y)

#Feature Importance
def importance(model):
    imp={}
    model.fit(X,y)
    importance=list(model.feature_importances_)
    for i in range(len(importance)):
        imp[importance[i]]=dataset.columns[i]
    return imp

importance(RandomForestClassifier())

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=50)

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
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier

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
    result=cross_val_score(models[i][0],X,y,cv=kfold)
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

# Ensemble models
estimators=[]
model1=LogisticRegression()
estimators.append(('logiscticr',model1))
model2=DecisionTreeClassifier()
estimators.append(('decision',model2))
model3=GaussianNB()
estimators.append(('svm',model3))

ensemble=VotingClassifier(estimators)
myresults=cross_val_score(ensemble,X,y,cv=kfold)
myresults.mean()

fitmodel(ensemble)

#Classification Report
from sklearn import metrics
from sklearn.metrics import classification_report

metrics.accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))

total1=sum(sum(cm))
accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)
sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )
specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)

# Visualising the Training set results
