from flask import Flask,session,render_template,redirect,request,url_for,g
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)


@app.before_request
def before_request():
	g.user = None
	if 'user' in session:
		g.user = session['user']


@app.route("/")
def index():
	#session['']
	return render_template('index.html')

@app.route("/login", methods=['GET','POST'])
def login():
	if request.method == 'POST':
		session.pop('user',None)
		#other varification section
		if request.form['password'] == 'asd':
			session['user'] = request.form['username']
			return redirect(url_for('protected'))
			#return render_template('protected.html')
	return render_template('login.html')

@app.route("/protected")
def protected():
	if g.user:
		return render_template('protected.html')
	return redirect(url_for('login'))


if  __name__ == '__main__':
	app.run(debug=True)