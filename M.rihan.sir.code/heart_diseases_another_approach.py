from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas
from keras.layers import Activation
from keras.layers import Dropout
from keras.optimizers import SGD
from sklearn.utils import shuffle
import csv

total_list = []

header_list = []
with open("data-835.csv", "r") as heart_disease_data:
    #     header_list = next(heart_disease_data)
    for line in heart_disease_data:
        data_list = line.split(',')
        temp_list = []
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
        total_list[idx][0] = float(total_list[idx][0]) / 200

    if (total_list[idx][1] == 'Male'):
        total_list[idx][1] = np.exp(0.8) / 10
    else:
        total_list[idx][1] = np.exp(0.6) / 10

    if (total_list[idx][2] == 'no'):
        total_list[idx][2] = np.exp(0.1) / 10
    elif (total_list[idx][2] == 'yes'):
        total_list[idx][2] = np.exp(0.5) / 10
    else:
        total_list[idx][2] = np.exp(1) / 10

    if (total_list[idx][3] == 'no'):
        total_list[idx][3] = np.exp(0.1) / 10
    elif (total_list[idx][3] == 'yes'):
        total_list[idx][3] = np.exp(1) / 10
    else:
        total_list[idx][3] = np.exp(0.5) / 10

    if (total_list[idx][4] == 'no'):
        total_list[idx][4] = np.exp(0.1) / 10
    elif (total_list[idx][4] == 'yes'):
        total_list[idx][4] = np.exp(1) / 10
    else:
        total_list[idx][4] = np.exp(0.5) / 10

    if (total_list[idx][5] == 'no'):
        total_list[idx][5] = np.exp(0.1) / 10
    elif (total_list[idx][5] == 'yes'):
        total_list[idx][5] = np.exp(1) / 10
    else:
        total_list[idx][5] = np.exp(0.5) / 10

    if (total_list[idx][6] == 'no'):
        total_list[idx][6] = np.exp(1) / 10
    elif (total_list[idx][6] == 'yes'):
        total_list[idx][6] = np.exp(0.1) / 10
    else:
        total_list[idx][6] = np.exp(0.5) / 10

    if (total_list[idx][7] == 'no'):
        total_list[idx][7] = np.exp(0.1) / 10
    elif (total_list[idx][7] == 'yes'):
        total_list[idx][7] = np.exp(1) / 10
    else:
        total_list[idx][7] = np.exp(0.5) / 10

    if (total_list[idx][8] == 'no'):
        total_list[idx][8] = np.exp(0.1) / 10
    elif (total_list[idx][8] == 'yes'):
        total_list[idx][8] = np.exp(1) / 10
    else:
        total_list[idx][8] = np.exp(0.5) / 10

    if (total_list[idx][9] == 'no'):
        total_list[idx][9] = np.exp(0.1) / 10
    elif (total_list[idx][9] == 'yes'):
        total_list[idx][9] = np.exp(1) / 10
    else:
        total_list[idx][9] = np.exp(0.5) / 10

    if (total_list[idx][10] == 'no'):
        total_list[idx][10] = np.exp(0.1) / 10
    elif (total_list[idx][10] == 'yes'):
        total_list[idx][10] = np.exp(1) / 10
    else:
        total_list[idx][10] = np.exp(0.5) / 10

    if (total_list[idx][11] == 'no'):
        total_list[idx][11] = np.exp(0.1) / 10
    elif (total_list[idx][11] == 'yes'):
        total_list[idx][11] = np.exp(1) / 10
    else:
        total_list[idx][11] = np.exp(0.5) / 10

    if (total_list[idx][12] == 'no'):
        total_list[idx][12] = np.exp(0.1) / 10
    elif (total_list[idx][12] == 'yes'):
        total_list[idx][12] = np.exp(1) / 10
    else:
        total_list[idx][12] = np.exp(0.5) / 10


    if((total_list[idx][13] == 'normal' and total_list[idx][14] == 'No') or (total_list[idx][13] == '?' and total_list[idx][14] == 'No')) :
        
        total_list[idx][16] = 'low'
        total_list[idx].append(1)
        total_list[idx].append(0)
        total_list[idx].append(0)
    elif((total_list[idx][13] == 'abnormal' and total_list[idx][14] == 'No') or (total_list[idx][13] == 'normal' and total_list[idx][14] == 'Yes')):
        
        total_list[idx][16] = 'medium'
        total_list[idx].append(0)
        total_list[idx].append(1)
        total_list[idx].append(0)
    else:
        
        total_list[idx][16] = 'high'
        total_list[idx].append(0)
        total_list[idx].append(0)
        total_list[idx].append(1)
        

with open('data.tab', 'w') as datafile:
    writer = csv.writer(datafile, delimiter='\t')
    for index in range(len(total_list)):
        writer.writerow(total_list[index])

dataframe = pandas.read_csv('data.tab', sep='\t')
dataset = dataframe.values
# print(dataset)
for lm in range(50):
    dataset = shuffle(dataset)
# print(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.70)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

print(len(train), len(test))

X_train = train[:,0:13]
Y_train = train[:,17:20]

X_test = test[:, 0:13]
Y_test = test[:,17:20]

num_epoch = 1500


# Create Model

model = Sequential()
model.add(Dense(32, input_shape=(13,) , activation= 'relu',init='uniform'))
model.add(Dropout(0.15))
model.add(Dense(32, activation= 'relu' ,init='uniform'))
model.add(Dropout(0.2))
model.add(Dense(3, activation = 'softmax'))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True)
# categorical_crossentropy
# 1e-6
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

real_ans = np.argmax(Y_test, axis=1)
temp_ans = np.argmax(res, axis=1)

count = 0
for l in range(len(real_ans)):
    if (temp_ans[l] == real_ans[l]):
        count = count + 1
print("Right Prediction Number: ", count)

count_actual_low = 0
count_actual_medium = 0
count_actual_high = 0

actual_low_index_list = []
actual_medium_index_list = []
actual_high_index_list = []

for m in range(len(real_ans)):
    if (real_ans[m] == 0):
        count_actual_low = count_actual_low + 1
        actual_low_index_list.append(m)
    elif (real_ans[m] == 1):
        count_actual_medium = count_actual_medium + 1
        actual_medium_index_list.append(m)
    else:
        count_actual_high = count_actual_high + 1
        actual_high_index_list.append(m)

print(count_actual_low, count_actual_medium, count_actual_high)

count_predicted_low = 0
count_predicted_medium = 0
count_predicted_high = 0

predicted_low_index_list = []
predicted_medium_index_list = []
predicted_high_index_list = []

for n in range(len(temp_ans)):
    if (temp_ans[n] == 0):
        count_predicted_low = count_predicted_low + 1
        predicted_low_index_list.append(n)
    elif (temp_ans[n] == 1):
        count_predicted_medium = count_predicted_medium + 1
        predicted_medium_index_list.append(n)
    else:
        count_predicted_high = count_predicted_high + 1
        predicted_high_index_list.append(n)

print(count_predicted_low, count_predicted_medium, count_predicted_high)

count_pure_extracted_low = 0

for o in range(len(predicted_low_index_list)):
    if predicted_low_index_list[o] in actual_low_index_list:
        count_pure_extracted_low = count_pure_extracted_low + 1

count_pure_extracted_medium = 0

for p in range(len(predicted_medium_index_list)):
    if predicted_medium_index_list[p] in actual_medium_index_list:
        count_pure_extracted_medium = count_pure_extracted_medium + 1

count_pure_extracted_high = 0

for q in range(len(predicted_high_index_list)):
    if predicted_high_index_list[q] in actual_high_index_list:
        count_pure_extracted_high = count_pure_extracted_high + 1

print(count_pure_extracted_low, count_pure_extracted_medium, count_pure_extracted_high)