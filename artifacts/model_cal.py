
from flask import Flask,request,render_template
from main import heart_disease

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data',methods = ['POST','GET'])
def get_data():
    data = request.form()