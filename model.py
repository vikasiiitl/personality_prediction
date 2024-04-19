import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('train dataset.csv')
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
input_cols = ['Gender', 'Age', 'openness', 'neuroticism', 'conscientiousness', 'agreeableness', 'extraversion']
output_cols = ['Personality (Class label)']

scaler = StandardScaler()
data[input_cols] = scaler.fit_transform(data[input_cols])
data.head()

X = data[input_cols]
Y = data[output_cols]
y=Y
test_data = pd.read_csv('test dataset.csv')
test_data['Gender'] = le.fit_transform(test_data['Gender'])
test_data[input_cols] = scaler.fit_transform(test_data[input_cols])
X_test = test_data[input_cols]
Y_test = test_data['Personality (class label)']
test_data.head()
def LR():
    

    model = LogisticRegression(multi_class='multinomial', solver='newton-cg',max_iter =1000)
    model.fit(X, Y)  

    

    y_pred= model.predict(X_test)  
    print("LR")
    print(accuracy_score(Y_test,y_pred)*100)
    print(y_pred)
def KNN():
    
    k=6
    classifier = KNeighborsClassifier(n_neighbors=k)

    classifier.fit(X, y)

    y_pred = classifier.predict(X_test)
    cmatrix = confusion_matrix(Y_test, y_pred)
# report = classification_report(y_test, y_pred, target_names=category_mapping.keys())
    print("KNN")
    print(accuracy_score(Y_test,y_pred)*100)
    print("Confusion Matrix:")
    print(cmatrix)
    print(y_pred)
def GNB():
    gnb = GaussianNB()
    gnb.fit(X, y)
    y_pred = gnb.predict(X_test)
    print('Naiye Bayes: {0:0.4f}'. format(accuracy_score(Y_test, y_pred)))
    print(y_pred)
LR()
KNN()
GNB()
#FINAL PREDICTION DONE BY VOTING ALL THREE MODELS CAN BE CHECKED IN APP.PY
