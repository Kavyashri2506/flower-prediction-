from flask import Flask,request,render_template, jsonify
import pickle
import numpy as np 

app = Flask(__name__)

#load the pickel model
model = pickle.load(open("model.pkl","rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods = ['post'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template('index.html', prediction_text = f'The flower species is {prediction}')

if __name__ == '__main__':
    app.run(debug=True)
