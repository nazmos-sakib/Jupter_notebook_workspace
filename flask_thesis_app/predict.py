from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template,flash,redirect,Response

from preprocess_data import prere

import numpy as np

#create instance of the flask
#"__name__" is the name of the application module
#ues this for single module

#app = Flask(__name__) 
app = Flask(__name__, static_url_path='')
#app.debug = True
#app.run(debug=True)

app.config['ENV'] = 'development'
app.config['DEBUG'] = True
app.config['TESTING'] = True

@app.route("/") # 
def running():
    return app.send_static_file('index.html')



@app.route('/hello',methods=["POST","GET"])

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

	x = []

	for key in message.keys():
		x.append(message[key])

	#print(x)

	user_data_process(x)
	prere()

	respone = {
	"gretting" : "hello, "+age+". your gender is : "+gender+". your smoking habit: "+smoking+". HTN: "+HTN+". DPL: "+DPL+". DM: "+DM+". physical_exercise: "+physical_exercise
	}

	return jsonify(respone)

def user_data_process(message):
	pass