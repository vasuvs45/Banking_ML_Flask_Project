from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import pickle

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
db = SQLAlchemy(app)

logistic_model  = pickle.load(open('C:/Users/vasuv/OneDrive/Desktop/Summer_2023/DataGlaciers/Project_Week/bank+marketing/bank/Deployment/ML_Model/logistic_model.pkl', 'rb'))
kmeans_model  = pickle.load(open('C:/Users/vasuv/OneDrive/Desktop/Summer_2023/DataGlaciers/Project_Week/bank+marketing/bank/Deployment/ML_Model/kmeans.pkl', 'rb'))
rfc_model  = pickle.load(open('C:/Users/vasuv/OneDrive/Desktop/Summer_2023/DataGlaciers/Project_Week/bank+marketing/bank/Deployment/ML_Model/rfc.pkl', 'rb'))
svm_model  = pickle.load(open('C:/Users/vasuv/OneDrive/Desktop/Summer_2023/DataGlaciers/Project_Week/bank+marketing/bank/Deployment/ML_Model/svm.pkl', 'rb'))

from ML_Model import routes