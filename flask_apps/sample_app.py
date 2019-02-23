from flask import Flask
#create instance of the flask
#"__name__" is the name of the application module
#ues this for single module
app = Flask(__name__) 

@app.route("/sample") # 

def running():
    return "flask is running"  