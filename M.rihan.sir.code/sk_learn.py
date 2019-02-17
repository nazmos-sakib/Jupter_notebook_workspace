import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB , BernoulliNB
from sklearn import svm
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as sk
import matplotlib.pyplot as plt

patient_data = pd.read_csv("IHD_data.csv")
patient_data = patient_data.sample(frac=1).reset_index(drop=True)

length = patient_data.shape[0]

patient_data['Naive Class'] = 0

for i in range(length):
    con1 = patient_data.loc[i, 'high'] == 1
    con2 = patient_data.loc[i, 'low'] == 1

    if con1:
        patient_data.loc[i, 'Naive Class'] = 2

    elif con2:
        patient_data.loc[i, 'Naive Class'] = 0

    else:
        patient_data.loc[i, 'Naive Class'] = 1

# patient_data.to_csv('naive_dataset.csv')

x_data = patient_data[
    ['Age', 'Sex', 'Smoking', 'HTN', 'DLP', 'DM', 'Physical Exercise', 'Family History', 'Drug History',
     'Psychological Stress', 'Chest Pain', 'Dyspnea', 'Palpitation']].replace([0, 1], [np.exp(0), np.exp(1)]).values
y_data = patient_data['Naive Class'].values



gnb = GaussianNB()
gnb.fit(x_data[:585],y_data[:585].ravel())
scr = gnb.score(x_data[586:],y_data[586:].ravel())

print('Naive Bayes: ',scr)
gnb.predict(x_data[586:])

svm_classifier = svm.SVC(decision_function_shape = 'ovo')
svm_classifier.fit(x_data[:585],y_data[:585].ravel())
accuracy = svm_classifier.score(x_data[586:],y_data[586:].ravel())

y_score = svm_classifier.predict(x_data[586:])
print(y_score)
print('SVM: ',accuracy)

regr = linear_model.LogisticRegression()
regr.fit(x_data[:585],y_data[:585].ravel())
rg_scr = regr.score(x_data[586:],y_data[586:].ravel())

print('regression: ',rg_scr)

dt_classifier = DecisionTreeClassifier(random_state=2)
dt_classifier.fit(x_data[:585],y_data[:585].ravel())
dt_scr = dt_classifier.score(x_data[586:],y_data[586:].ravel())

print('decision Tree: ',dt_scr)