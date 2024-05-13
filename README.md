# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: VIGNESH R
RegisterNumber:  212222230172
*/
```
```

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df.head()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['v2'])
y = df['v1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = svm.SVC (kernel='linear') 
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy: ", accuracy_score (y_test, predictions)) 
print("Classification Report: ")
print(classification_report (y_test, predictions))
```

## Output:

## Head() :

![328008742-37e16030-4437-49d8-b3ec-3f3c68a7d5fa](https://github.com/AdhithiyanK/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121029258/69fb79b4-5433-4c94-85a5-6a4f2d2ee46d)

## Kernel Model:

![328008775-d4e22103-87c3-43d4-a5e6-85fc8aaa8d7e](https://github.com/AdhithiyanK/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121029258/8d16d619-93c8-4832-91f8-fac46834f828)

## Accuracy and Classification Report :

![328008809-03a2362a-acf7-4f86-b310-309aca2446d6](https://github.com/AdhithiyanK/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121029258/eba4ae16-ed91-44cf-87e3-b994fdd5c057)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
