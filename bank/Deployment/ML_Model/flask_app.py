from flask import Flask, render_template, send_file
import pickle
import matplotlib.pyplot as plt
import io
import pandas as pd
import threading

app = Flask(__name__)
logistic_model = pickle.load(open('logistic_model.pkl', 'rb'))

def generate_pie_plot(predictions):
    plt.figure(figsize=(4, 4))
    num_yes = sum(predictions)
    num_no = len(predictions) - num_yes

    labels = ['Yes', 'No']
    sizes = [num_yes, num_no]
    colors = ['lightcoral', 'lightskyblue']
    explode = (0.1, 0)

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.title("Customer Subscription Prediction")

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return buffer

def generate_bar_plot(predictions):
    plt.figure(figsize=(4, 4))
    num_yes = sum(predictions)
    num_no = len(predictions) - num_yes

    labels = ['Yes', 'No']
    counts = [num_yes, num_no]

    plt.bar(labels, counts, color=['lightcoral', 'lightskyblue'])
    plt.xlabel('Prediction')
    plt.ylabel('Count')
    plt.title('Customer Subscription Prediction')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return buffer

def send_plot_buffer(plot_buffer):
    with app.app_context():
        response = send_file(plot_buffer, mimetype='image/png')
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/plot')
def plot():
    return render_template('plot.html')

@app.route('/generate_pie_chart')
def generate_pie_chart():
    input_data = pd.read_csv('oversampled_train.csv')
    X = input_data[['age', 'job', 'marital', 'education', 'balance', 'housing', 'contact', 'day', 'duration', 'campaign', 'pdays', 'previous', 'y']]
    y = input_data['y']
    predictions = logistic_model.predict(X)

    plot_buffer = generate_pie_plot(predictions)
    return send_plot_buffer(plot_buffer)

@app.route('/generate_bar_chart')
def generate_bar_chart():
    input_data = pd.read_csv('oversampled_train.csv')
    X = input_data[['age', 'job', 'marital', 'education', 'balance', 'housing', 'contact', 'day', 'duration', 'campaign', 'pdays', 'previous', 'y']]
    y = input_data['y']
    predictions = logistic_model.predict(X)

    plot_buffer = generate_bar_plot(predictions)
    return send_plot_buffer(plot_buffer)

if __name__ == "__main__":
    app.run(debug=True)
