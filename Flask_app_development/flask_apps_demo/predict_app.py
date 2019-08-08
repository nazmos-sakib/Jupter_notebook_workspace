#import base64
import io

#import all necessary liberty

import tensorflow as tf
import numpy as np

import pandas as pd


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras import backend as k 


import csv

#spliting dataset into traning set and test set
from sklearn.model_selection import train_test_split
#missing value handle
from sklearn.preprocessing import Imputer
#To shuffle the data set
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
#from sklearn.utils import shuffle

import itertools

from scipy.stats import zscore

import os

from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)

def get_model():
	global model
	model = load_model("heart_attack_risk_prediction_fold_no_8_with_cross_validation.h5")
	print("* Model loaded")

def preprocesso_data():
	pass

print("* Loading keras model....")
get_model()

@app.route("/predict",methods=["POST"])

def predict():
	message = request.get_json(force=True)
	name = message['name']

	respone = {
	"gretting" : "hello, "+name
	}

	return jsonify(respone)
