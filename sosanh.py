import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron


dataset = pd.read_csv('diabetes.csv')
x = dataset.drop("Outcome",axis=1)
y = dataset["Outcome"]

train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15

# train is now 75% of the entire data set
# the _junk suffix means that we drop that variable completely
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio)

# test is now 15% of the initial data set
# validation is now 15% of the initial data set
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

print(x_train, x_val, x_test)

models = {
          "Logistic Regression": LogisticRegression(), 
          "Decision Tree": DecisionTreeClassifier(),
          "Naive Bayes": GaussianNB(),
          "Perceptron Neutral network": Perceptron()
          }

def fit_and_score(models, x_train, x_test, y_train, y_test):
    np.random.seed(42)
    model_scores_test = {}
    model_scores_val = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        model_scores_test[name] = model.score(x_test, y_test)*100
        model_scores_val[name] = model.score(x_val, y_val)*100
    return model_scores_test,model_scores_val


model_scores = fit_and_score(models=models,x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test)

print('Kết quả thực hiện trên tập test')
model_scores[0]

print('Kết quả thực hiện trên tập validation')
model_scores[1]

model= LogisticRegression()
model1=DecisionTreeClassifier()
model2=GaussianNB()
model3=Perceptron()

model.fit(x_train,y_train)
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)


y_pred_test1 = model.predict(x_test)
y_pred_test2 = model1.predict(x_test)
y_pred_test3 = model2.predict(x_test)
y_pred_test4 = model3.predict(x_test)
y_pred_val1 = model.predict(x_val)
y_pred_val2 = model1.predict(x_val)
y_pred_val3 = model2.predict(x_val)
y_pred_val4 = model3.predict(x_val)

print('Ket qua hoi quy tuyen tinh tren tap test')
plot_confusion_matrix(model,x_test,y_test)
print('precision_score :',precision_score(y_test, y_pred_test1, average="macro"))
print('recall_score :',recall_score(y_test, y_pred_test1, average="macro")) 
print('f1_score :',f1_score(y_test, y_pred_test1, average="macro"))

print('Ket qua hoi quy tuyen tinh tren tap Validation')
plot_confusion_matrix(model,x_val,y_val)
print('precision_score :',precision_score(y_val, y_pred_val1, average="macro"))
print('recall_score :',recall_score(y_val, y_pred_val1, average="macro")) 
print('f1_score :',f1_score(y_val, y_pred_val1, average="macro"))

print('Ket qua Cay quyet tinh tren tap Test')
plot_confusion_matrix(model1,x_test,y_test)
print('precision_score :',precision_score(y_test, y_pred_test2, average="macro"))
print('recall_score :',recall_score(y_test, y_pred_test2, average="macro")) 
print('f1_score :',f1_score(y_test, y_pred_test2, average="macro"))

print('Ket qua Cay quyet tinh tren tap Validation')
plot_confusion_matrix(model1,x_val,y_val)
print('precision_score :',precision_score(y_val, y_pred_val2, average="macro"))
print('recall_score :',recall_score(y_val, y_pred_val2, average="macro")) 
print('f1_score :',f1_score(y_val, y_pred_val2, average="macro"))

print('Ket qua Bayes tho ngay tren tap Test')
plot_confusion_matrix(model2,x_test,y_test)
print('precision_score :',precision_score(y_test, y_pred_test3, average="macro"))
print('recall_score :',recall_score(y_test, y_pred_test3, average="macro")) 
print('f1_score :',f1_score(y_test, y_pred_test3, average="macro"))

print('Ket qua Bayes tho ngay tren tap Validation')
plot_confusion_matrix(model2,x_val,y_val)
print('precision_score :',precision_score(y_val, y_pred_val3, average="macro"))
print('recall_score :',recall_score(y_val, y_pred_val3, average="macro")) 
print('f1_score :',f1_score(y_val, y_pred_val3, average="macro"))

print('Ket qua Mang no ron nhan tao tren tap Test')
plot_confusion_matrix(model3,x_test,y_test)
print('precision_score :',precision_score(y_test, y_pred_test4, average="macro"))
print('recall_score :',recall_score(y_test, y_pred_test4, average="macro")) 
print('f1_score :',f1_score(y_test, y_pred_test4, average="macro"))

print('Ket qua Mang no ron nhan tao tren tap Validation')
plot_confusion_matrix(model3,x_val,y_val)
print('precision_score :',precision_score(y_val, y_pred_val4, average="macro"))
print('recall_score :',recall_score(y_val, y_pred_val4, average="macro")) 
print('f1_score :',f1_score(y_val, y_pred_val4, average="macro"))