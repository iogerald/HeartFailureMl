import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd 

app=Flask(__name__)
## load the model
rfmodel=pickle.load(open('rfmodel.pkl','rb'))
std=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=std.transform(np.array(list(data.values())).reshape(1,-1))
    output=rfmodel.predict(new_data)
    print(output[0])
    return jsonify(int(output[0]))

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=std.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=rfmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text="The stroke occurrence is {}".format(output))

if __name__=="__main__":
    with app.app_context():
        app.run(debug=True)