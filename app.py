from flask import Flask, render_template,request
import pickle
import numpy as np

model=pickle.load(open('iriss.pkl','rb'))

app = Flask(__name__, template_folder='template')


@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def index():
    data1=data2=data3=data4=1.0
    data1=request.form['a']
    data2=request.form['b']
    data3=request.form['c']
    data4=request.form['d']
    arr=np.array([[data1,data2,data3,data4]])
    print(arr)
    pred=model.predict(arr)
    return render_template('predicted.html', data=pred)
    #return "hello"

if __name__ == "__main__":
    app.run(debug=True)