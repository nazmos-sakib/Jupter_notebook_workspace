
from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template,flash,redirect,Response

from preprocess_data import user_data_process
from model import get_model,NN,NN1

import numpy as np
import pandas as pd

import yaml




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




#database
from flask_mysqldb import MySQL
#from flaskext.mysql import MySQL

db = yaml.load(open('db.yaml'))

# DB configuration
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']

mysql = MySQL(app)



#model1 = get_model()
from keras.models import load_model
#print ('load model...')
model = load_model("heart_attack_risk_prediction_percent_split.h5",compile= False)
model._make_predict_function()
#print("* Model loaded")


def NN2(data):
    data = np.expand_dims(data, axis=0)
    #global model
    predections = model.predict_classes(data,verbose=0)
    return predections[0]



#print("------------the begin print----")

@app.route("/") # 
def running():
	#print("--in route fun--")
	return app.send_static_file('index.html')

@app.route("/mhealthasd") # 
def mhealth():
	return app.send_static_file('predict.html')



@app.route('/hello',methods=["POST"])

def hello():
	print("--in hello fun--")
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
	#print(data)
	
	x = []

	for key in data.keys():
		x.append(data[key])

	#print(x)
	
	#predicted_value = []
	
	#predicted_value = NN(x)
	predicted_value = NN2(x)
	#predicted_value = NN1(x,model1)


	# Database insert
	cur = mysql.connection.cursor()
	cur.execute("INSERT INTO dataset(age,gender,smoking,HTN,DPL,DM,physical_exercise,family_history,drug_history,psychological_stress,chest_pain,dyspnea,palpitation,ECG,IHD) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(age,gender,smoking,HTN,DPL,DM,physical_exercise,family_history,drug_history,psychological_stress,chest_pain,dyspnea,palpitation,ECG,predicted_value))
	mysql.connection.commit()
	cur.close()

	result = ''
	if(predicted_value==1):
		result = 'According to the information our prediction result is YES'
	else:
		result = 'According to the information our prediction result is no'

	respone = {
	"gretting" : "hello,  Lets take a look at the information that you provide. your age is: "+age+". your gender is : "+gender+". your smoking habit: "+smoking+". HTN: "+HTN+". DPL: "+DPL+". DM: "+DM+". physical_exercise: "+physical_exercise+". your family_history: "+family_history+". drug_history: "+drug_history+". psychological_stress: "+psychological_stress+". chest_pain: "+chest_pain+". dyspnea: "+dyspnea+". palpitation: "+palpitation+". ECG Report: "+ECG+". ",
	"prediction" : result
	}

	return jsonify(respone)
	#return app.send_static_file('signup.html')



@app.route("/login") # 
def login():
	return app.send_static_file('login.html')

@app.route("/loginaction",methods=["POST"]) # 
def loginaction():
	data = request.get_json(force=True)

	email = data['email']
	password = data['password']

	cur = mysql.connection.cursor()

	# Database check email
	qresult = cur.execute("""select email from member where email = %s and password = %s""",(email,password))
	qfresult = cur.fetchone()
	#if len(qfresult) is 1:
	#	do something
	#
	print(type(qresult))
	if qresult <= 0:
		respone = {
		"flag" : 'no'
		}
		mysql.connection.commit()
		cur.close()
		return jsonify(respone)

	#cur.execute("INSERT INTO member(email,first_name,last_name,password) VALUES(%s,%s,%s,%s)",(email,first_name,last_name,password))

	mysql.connection.commit()
	cur.close()

	respone = {
		"flag" : 'yes'
	}
	return jsonify(respone)


@app.route("/signup") # 
def signup():
	return app.send_static_file('signup.html')

@app.route("/signupaction",methods=["POST"]) # 
def signupaction():
	data = request.get_json(force=True)

	email = data['email']
	first_name = data['first_name']
	last_name = data['last_name']
	password = data['password']

	cur = mysql.connection.cursor()

	# Database check email
	qresult = cur.execute("""select email from member where email = %s""",(email,))
	if qresult > 0:
		respone = {
		"flag" : 'no'
		}
		mysql.connection.commit()
		cur.close()
		return jsonify(respone)

	cur.execute("INSERT INTO member(email,first_name,last_name,password) VALUES(%s,%s,%s,%s)",(email,first_name,last_name,password))

	mysql.connection.commit()
	cur.close()

	respone = {
		"flag" : 'yes'
	}
	return jsonify(respone)


if __name__ == "__main__":
	app.run(host='0.0.0.0')
