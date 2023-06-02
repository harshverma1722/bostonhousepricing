import pickle
from flask import Flask,request,app, jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
### loading the model
regmodel = pickle.load(open("regmodel.pkl",'rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route("/predict_api",methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1)) ## to extract the values from json and convert it into list and reshape it to 1 row and 13 columns
    new_data=scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


if __name__=="__main_":
    app.run(debug=True)


