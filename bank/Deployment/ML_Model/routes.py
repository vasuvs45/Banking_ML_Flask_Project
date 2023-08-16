from flask import Flask, request,render_template,send_file,url_for, flash, redirect 
from ML_Model import app, logistic_model,kmeans_model,rfc_model,svm_model
from ML_Model.Methods import  generate_bar_plot
import pandas as pd
from ML_Model.form import RegistrationForm, LoginForm

@app.route('/')
def home():
    return render_template('index.html')


@app.route("/register", methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for('home'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if form.email.data == 'admin@blog.com' and form.password.data == 'password':
            flash('You have been logged in!', 'success')
            return redirect(url_for('First_Page'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route('/First_Page')
def First_Page():
    return render_template('First_Page.html')


# @app.route('/plot')
# def plot():
#     return render_template('plot.html')


# @app.route('/generate_pie_chart')
# def generate_pie_chart():
#     input_data = pd.read_csv('C:/Users/vasuv/OneDrive/Desktop/Summer_2023/DataGlaciers/Project_Week/bank+marketing/bank/Deployment/ML_Model/oversampled_train.csv')
#     X = input_data[['age', 'job', 'marital', 'education', 'balance', 'housing', 'contact', 'day', 'duration','campaign', 'pdays', 'previous', 'y']]
#     y = input_data['y']
#     predictions = logistic_model.predict(X)
#     plot_buffer = generate_pie_plot(predictions)

#     return send_file(plot_buffer, mimetype='image/png')

@app.route('/generate_logistic_bar_chart')
def generate_logistic_bar_chart():
    input_data = pd.read_csv('C:/Users/vasuv/OneDrive/Desktop/Summer_2023/DataGlaciers/Project_Week/bank+marketing/bank/Deployment/ML_Model/oversampled_train.csv')
    X = input_data[['age', 'job', 'marital', 'education', 'balance', 'housing', 'contact', 'day', 'duration','campaign', 'pdays', 'previous', 'y']]
    y = input_data['y']
    predictions = logistic_model.predict(X)
    plot_buffer = generate_bar_plot(predictions)

    return send_file(plot_buffer, mimetype='image/png')

@app.route('/generate_k_means_bar_chart')
def generate_k_means_bar_chart():
    input_data = pd.read_csv('C:/Users/vasuv/OneDrive/Desktop/Summer_2023/DataGlaciers/Project_Week/bank+marketing/bank/Deployment/ML_Model/oversampled_train.csv')
    X = input_data[['age', 'job', 'marital', 'education', 'balance', 'housing', 'contact', 'day', 'duration','campaign', 'pdays', 'previous', 'y']]
    y = input_data['y']
    predictions = kmeans_model.predict(X)
    plot_buffer = generate_bar_plot(predictions)

    return send_file(plot_buffer, mimetype='image/png')

@app.route('/generate_random_forest_chart')
def generate_random_forest_chart():
    input_data = pd.read_csv('C:/Users/vasuv/OneDrive/Desktop/Summer_2023/DataGlaciers/Project_Week/bank+marketing/bank/Deployment/ML_Model/oversampled_train.csv')
    X = input_data[['age', 'job', 'marital', 'education', 'balance', 'housing', 'contact', 'day', 'duration','campaign', 'pdays', 'previous', 'y']]
    y = input_data['y']
    predictions = rfc_model.predict(X)
    plot_buffer = generate_bar_plot(predictions)

    return send_file(plot_buffer, mimetype='image/png')

@app.route('/generate_svm_chart')
def generate_svm_chart():
    input_data = pd.read_csv('C:/Users/vasuv/OneDrive/Desktop/Summer_2023/DataGlaciers/Project_Week/bank+marketing/bank/Deployment/ML_Model/oversampled_train.csv')
    X = input_data[['age', 'job', 'marital', 'education', 'balance', 'housing', 'contact', 'day', 'duration','campaign', 'pdays', 'previous', 'y']]
    y = input_data['y']
    predictions = svm_model.predict(X)
    plot_buffer = generate_bar_plot(predictions)

    return send_file(plot_buffer, mimetype='image/png')
