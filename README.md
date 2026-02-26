# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages. 

2.Import the dataset to operate on. 

3.Split the dataset.

4.Predict the required output.

5.End the program.

## Program:

/*
Program to implement the SVM For Spam Mail Detection..
Developed by: shalini.N
RegisterNumber:  212224040305
*/
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:

<img width="819" height="283" alt="image" src="https://github.com/user-attachments/assets/98f84313-ebfa-4ed5-a672-9437edccad38" />

<img width="504" height="281" alt="image" src="https://github.com/user-attachments/assets/60d69f58-8380-401e-a1ec-b953d1bb4278" />

<img width="218" height="346" alt="image" src="https://github.com/user-attachments/assets/f92d9fe2-3c9d-4fbd-b2ff-318a73384d25" />

<img width="718" height="435" alt="image" src="https://github.com/user-attachments/assets/81ae9f1b-2cc9-443d-9da0-7c34cc447be0" />

<img width="258" height="42" alt="image" src="https://github.com/user-attachments/assets/1c2669ed-392a-4ba9-9dbb-564c942c86dd" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
