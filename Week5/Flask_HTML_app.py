import numpy as np
from flask import Flask,request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('linear_regression_model.pickle','rb'))

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    #output = round(prediction[0])
    return render_template('index.html',prediction_text = 'Salary Should be  $ {}'.format(prediction))

if __name__ == "__main__":
    app.run(port = 5000, debug = True)