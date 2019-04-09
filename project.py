import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
 
###

dataset = pd.read_csv('heart.csv')
dataset.info()
dataset.isnull().sum()
per_missing_values =  100 * dataset.isnull().sum() / len(dataset)
dataset.describe()

###

X = dataset.iloc[:, 0:14].values
y = dataset.iloc[:, -1].values

###

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func = chi2, k= 10)
fit = bestfeatures.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(dataset.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis= 1)
featureScores.columns = ['Specs', 'Score']
print(featureScores.nlargest(10, 'Score'))

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X, y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index = X.columns)
feat_importances.nlargest(10).plot(kind ='barh')
plt.show()

corrmat = dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")

###

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelx= LabelEncoder()
X[:, 0] = labelx.fit_transform(X[:, 0])

onehotx = OneHotEncoder(categorical_features = [0])
X = onehotx.fit_transform(X).toarray()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
###dataset = pd.get_dummies(dataset)###

###

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

###

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

scores = []

def cross_val(model, X_train, y_train):
    val_scores = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
    scores.append([model, val_scores.mean(), val_scores.std()])
    
for i in range(0, 6):
    cross_val(models[i],X_train, y_train)
    
###

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

models = [RandomForestClassifier(n_estimators = 10, criterion= 'entropy'), 
          DecisionTreeClassifier(), 
          SVC(), 
          GaussianNB(), 
          LogisticRegression(),
          XGBClassifier()
          ]

def model_report(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(model, classification_report(y_test, y_pred))

model_report(models[1], X_train, X_test, y_train, y_test)















