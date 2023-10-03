from flask import Flask, request, jsonify
import numpy as np
import pickle

#creating a Flask app
app = Flask(__name__)

@app.route('/',methods = ['GET','POST'])

def home():
    if(request.method == 'GET'):

        data="hello world"
        return jsonify({'data' : data})

with open('linear_regression_model.pickle', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the input data from the request

    # Preprocess the input data if necessary
    years_experience = np.array(data['years_experience']).reshape(-1, 1)

    # Make predictions using the loaded model
    predicted_salary = model.predict(years_experience)

    # Return the predicted result as JSON
    response = {'predicted_salary': predicted_salary.tolist()}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
