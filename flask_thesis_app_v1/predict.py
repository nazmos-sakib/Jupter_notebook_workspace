from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template,flash,redirect,Response

from preprocess_data import user_data_process
from model import get_model,NN,NN1

import numpy as np
import pandas as pd


'''
import tensorflow as tf

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
'''



#create instance of the flask
#"__name__" is the name of the application module
#ues this for single module

#app = Flask(__name__) 
app = Flask(__name__, static_url_path='')
#app.debug = True
#app.run(debug=True)

#app.config['ENV'] = 'development'
#app.config['DEBUG'] = True
#app.config['TESTING'] = True


#model1 = get_model()
from keras.models import load_model
print ('load model...')
model = load_model("heart_attack_risk_prediction_percent_split.h5",compile= False)
model._make_predict_function()
print("* Model loaded")


def NN2(data):
    data = np.expand_dims(data, axis=0)
    #global model
    predections = model.predict_classes(data,verbose=0)
    return predections[0]



print("------------the begin print----")

@app.route("/") # 
def running():
	print("--in route fun--")
	return app.send_static_file('index.html')



@app.route('/hello',methods=["POST"])

def hello():
	message = request.get_json(force=True)
	age = message['age']
	gender = message['gender']
	smoking = message['smoking']
	HTN = message['HTN']
	DPL = message['DPL']
	DM = message['DM']
	physical_exercise = message['physical_exercise']
	family_history = message['family_history']
	drug_history = message['drug_history']
	psychological_stress = message['psychological_stress']
	chest_pain = message['chest_pain']
	dyspnea = message['dyspnea']
	palpitation = message['palpitation']
	ECG = message['ECG']

	"""for key in message:
					print(message[key])
				
				asd = message.values
				asd = np.array(asd)
				for i in len(asd):
					print(asd[i])"""

	#print(type(asd))

	#x = []

	#for key in message.keys():
	#	x.append(message[key])

	#print(x)

	data = user_data_process(message)
	print(data)
	
	x = []

	for key in data.keys():
		x.append(data[key])

	print(x)
	
	#predicted_value = []
	
	#predicted_value = NN(x)
	predicted_value = NN2(x)
	#predicted_value = NN1(x,model1)

	respone = {
	"gretting" : "hello, "+age+". your gender is : "+gender+". your smoking habit: "+smoking+". HTN: "+HTN+". DPL: "+DPL+". DM: "+DM+". physical_exercise: "+physical_exercise,
	"prediction" : str(predicted_value)
	}

	return jsonify(respone)

