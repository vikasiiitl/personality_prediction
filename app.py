import numpy as np
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
app = Flask(__name__)
# model = joblib.load("train_model.pkl")
scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      gender = request.form['gender']
      if(gender == "Female"):
        gender_no = 1
      else:
        gender_no = 2
      age = request.form['age']
      openness = request.form['openness']
      neuroticism = request.form['neuroticism']
      conscientiousness = request.form['conscientiousness']
      agreeableness = request.form['agreeableness']
      extraversion = request.form['extraversion']
      result = np.array([gender_no, age, openness,neuroticism, conscientiousness, agreeableness, extraversion], ndmin = 2)
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

      model = LogisticRegression(multi_class='multinomial', solver='newton-cg',max_iter =1000)
      model.fit(X, Y) 
      k=6
      classifier = KNeighborsClassifier(n_neighbors=k)

      classifier.fit(X, Y)
      gnb = GaussianNB()
      gnb.fit(X, Y) 

      final = [np.float64(x) for x in result]
      # personality = str(model.predict(final)[0])
      personality1 = str(gnb.predict(final)[0])
      personality2=str(classifier.predict(final)[0])
      personality3=str(model.predict(final)[0])
      print(personality3)
      print(personality1)
      print(personality2)
      if personality1==personality3:
        return render_template("submit.html",answer = personality1)
      elif personality1==personality2:
        return render_template("submit.html",answer = personality1)
      elif personality3==personality2:
         return render_template("submit.html",answer = personality2)
      else:
         return render_template("submit.html",answer=personality3)

if __name__ == '__main__':
    app.run()
