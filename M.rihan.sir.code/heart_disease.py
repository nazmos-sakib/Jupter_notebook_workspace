from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas
from keras.layers import Activation
from keras.layers import Dropout
from keras.optimizers import SGD
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import csv

total_list =[]

header_list = []
with open("data-835.csv","r") as heart_disease_data:
#     header_list = next(heart_disease_data)
    for line in heart_disease_data:
        data_list = line.split(',')
        temp_list =[]
        for k in range(len(data_list)):
            single_val = data_list[k]
            single_val = single_val.rstrip("\n")
            temp_list.append(single_val)
        total_list.append(temp_list)

        
total_list[0].append('low')
total_list[0].append('medium')
total_list[0].append('high')


for idx in range(1, len(total_list)):

    if (total_list[idx][0] == 'NA' or total_list[idx][0] == '?'):
        total_list[idx][0] = np.random.uniform(0.25, 0.8)
    else:
        total_list[idx][0] = float(total_list[idx][0]) / 50

    if (total_list[idx][1] == 'Male'):
        total_list[idx][1] = np.exp(0.8) 
    else:
        total_list[idx][1] = np.exp(0.6) 

    if (total_list[idx][2] == 'no'):
        total_list[idx][2] = np.exp(0.1) 
    elif (total_list[idx][2] == 'yes'):
        total_list[idx][2] = np.exp(0.5) 
    else:
        total_list[idx][2] = np.exp(1) 

    if (total_list[idx][3] == 'no'):
        total_list[idx][3] = np.exp(0.1) 
    elif (total_list[idx][3] == 'yes'):
        total_list[idx][3] = np.exp(1) 
    else:
        total_list[idx][3] = np.exp(0.5) 

    if (total_list[idx][4] == 'no'):
        total_list[idx][4] = np.exp(0.1) 
    elif (total_list[idx][4] == 'yes'):
        total_list[idx][4] = np.exp(1) 
    else:
        total_list[idx][4] = np.exp(0.5) 

    if (total_list[idx][5] == 'no'):
        total_list[idx][5] = np.exp(0.1) 
    elif (total_list[idx][5] == 'yes'):
        total_list[idx][5] = np.exp(1) 
    else:
        total_list[idx][5] = np.exp(0.5) 

    if (total_list[idx][6] == 'no'):
        total_list[idx][6] = np.exp(1) 
    elif (total_list[idx][6] == 'yes'):
        total_list[idx][6] = np.exp(0.1) 
    else:
        total_list[idx][6] = np.exp(0.5) 

    if (total_list[idx][7] == 'no'):
        total_list[idx][7] = np.exp(0.1) 
        total_list[idx][7] = np.exp(1) 
    else:
        total_list[idx][7] = np.exp(0.5) 

    if (total_list[idx][8] == 'no'):
        total_list[idx][8] = np.exp(0.1) 
    elif (total_list[idx][8] == 'yes'):
        total_list[idx][8] = np.exp(1) 
    else:
        total_list[idx][8] = np.exp(0.5) 

    if (total_list[idx][9] == 'no'):
        total_list[idx][9] = np.exp(0.1) 
    elif (total_list[idx][9] == 'yes'):
        total_list[idx][9] = np.exp(1) 
    else:
        total_list[idx][9] = np.exp(0.5) 

    if (total_list[idx][10] == 'no'):
        total_list[idx][10] = np.exp(0.1) 
    elif (total_list[idx][10] == 'yes'):
        total_list[idx][10] = np.exp(1) 
    else:
        total_list[idx][10] = np.exp(0.5) 
    if (total_list[idx][11] == 'no'):
        total_list[idx][11] = np.exp(0.1) 
    elif (total_list[idx][11] == 'yes'):
        total_list[idx][11] = np.exp(1) 
    else:
        total_list[idx][11] = np.exp(0.5) 

    if (total_list[idx][12] == 'no'):
        total_list[idx][12] = np.exp(0.1) 
    elif (total_list[idx][12] == 'yes'):
        total_list[idx][12] = np.exp(1) 
    else:
        total_list[idx][12] = np.exp(0.5) 

    if (total_list[idx][13] == 'normal'):
        total_list[idx][13] = np.exp(1)
    elif (total_list[idx][13] == 'abnormal'):
        total_list[idx][13] = np.exp(0.1)
    else:
        total_list[idx][13] = np.exp(0.5)

    if (total_list[idx][14] == 'No'):
        total_list[idx][14] = 0
    elif (total_list[idx][14] == 'Yes'):
        total_list[idx][14] = 1
    else:
        total_list[idx][14] = 0.5

    if (total_list[idx][16] == 'low'):
        total_list[idx].append(1)
        total_list[idx].append(0)
        total_list[idx].append(0)
        #total_list[idx][16] = [1,0,0]
    elif (total_list[idx][16] == 'high'):
        total_list[idx].append(0)
        total_list[idx].append(0)
        total_list[idx].append(1)
        #total_list[idx][16] = [0, 0, 1]
    else:
        total_list[idx].append(0)
        total_list[idx].append(1)
        total_list[idx].append(0)
        #total_list[idx][16] = [0, 1, 0]



with open('data.tab', 'w') as datafile:
    writer = csv.writer(datafile, delimiter='\t')
    for index in range(len(total_list)):
        writer.writerow(total_list[index])
        
with open('IHD_data.csv', 'w') as datafile:
    writer = csv.writer(datafile, delimiter=',')
    for index in range(len(total_list)):
        writer.writerow(total_list[index])

dataframe = pandas.read_csv('data.tab', sep='\t')
dataset = dataframe.values


for lm in range(50):
    dataset = shuffle(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.70)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

print(len(train), len(test))


X_train = train[:,0:14]
Y_train = train[:,14]

X_test = test[:,0:14]
Y_test = test[:,14]



num_epoch = 300

# Create Model

model = Sequential()
model.add(Dense(16, input_shape=(14,) , activation= 'sigmoid',init='uniform'))
model.add(Dropout(0.1))
#model.add(Dense(64, activation= 'sigmoid' ,init='uniform'))
#model.add(Dropout(0.15))
model.add(Dense(1 , activation = 'sigmoid'))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#categorical_crossentropy
#1e-6
model.compile(loss='binary_crossentropy',
                  optimizer= sgd,
                  metrics=['accuracy', 'f1score', 'precision', 'recall'])
model.summary()



model.fit(X_train, Y_train, nb_epoch=num_epoch , verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test , Y_test , verbose=0)

print('\n')
print('Loss:', score[0])
print('Test accuracy:', score[1])
print('F1 Score:', score[2])
print('Precision:', score[3])
print('Recall:', score[4])

res = model.predict(X_test)
predict = []
for pred_idx in range(len(res)):
    if(res[pred_idx] > 0.7):
        predict.append(1)
    else:
        predict.append(0)





true_positive = 0
true_negative = 0 
false_positive = 0
false_negative = 0


for l in range(len(res)):
    if(predict[l] == 1 and  Y_test[l] == 1):
        true_positive = true_positive + 1
    elif(predict[l] == 0 and Y_test[l] == 0):
        true_negative = true_negative + 1
    elif(predict[l] == 1 and  Y_test[l] == 0):
        false_positive = false_positive + 1
    elif(predict[l] == 0 and  Y_test[l] == 1):
        false_negative = false_negative + 1


    


print("True Positive: ",true_positive )
print("True Negative: ",true_negative )
print("False Positive: ",false_positive )
print("False Negative: ",false_negative )

specifity = 0
tot_specifity = (true_negative + false_positive)
print(' TSP: ', tot_specifity)
specifity = true_negative / tot_specifity
print('Specifity: ', specifity)

sensitivity = 0
tot_sensitivity = (true_positive + false_negative)
print(' TSN: ', tot_sensitivity)
sensitivity = true_positive / tot_sensitivity
print('Sensitivity: ' ,sensitivity)

fpr = dict()
tpr = dict()

roc_auc = dict()

fpr , tpr , _ = roc_curve(Y_test[:], res[:])
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()





