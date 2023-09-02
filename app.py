import pandas as pd
import numpy as np
from flask import Flask, request, render_template #render_template help in getting back to home page.
import pickle


label_encoder = pickle.load(open('models/label_encoder.pkl','rb'))
model = pickle.load(open('models/Customer_churn_model.pkl','rb'))
minmaxscaler = pickle.load(open('models/minmaxscaler.pkl','rb'))
genders_ = ["Male", "Female"]
locations_ = ['Los Angeles', 'New York', 'Miami', 'Chicago', 'Houston']

app = Flask(__name__, template_folder='templates')

def predictions(data_dict):
     df = pd.DataFrame(data_dict)
     df[['Location','Gender']] = df[['Location','Gender']].apply(lambda x: label_encoder.fit_transform(x))
     df = minmaxscaler.transform(df)
     print(df)
     pred = model.predict(df)
     print(pred,'\n',pred[0])
     return pred[0]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
     if request.method == 'POST':
          to_predict_list = request.form.to_dict()
          Age = int(to_predict_list['Age'])
          Gender = to_predict_list['Gender']
          Location = to_predict_list['Location']
          Subscription_Length_Months = float(to_predict_list['Subscription_Length_Months'])
          Monthly_Bill = float(to_predict_list['Monthly_Bill'])
          Total_Usage_GB = float(to_predict_list['Total_Usage_GB'])
          print(Age,Gender,Location,Subscription_Length_Months,Monthly_Bill,Total_Usage_GB)

          data_dict = {
          'Age': [Age],
          'Gender': [Gender],
          'Location': [Location],
          'Subscription_Length_Months': [Subscription_Length_Months],
          'Monthly_Bill': [Monthly_Bill],
          'Total_Usage_GB': [Total_Usage_GB]
          }

          result = predictions(data_dict)
     return render_template('index.html', 
                            age=Age,
                            gender=Gender,
                            location=Location,
                            subscription_Length_Months=Subscription_Length_Months,
                            monthly_Bill=Monthly_Bill,
                            total_Usage_GB=Total_Usage_GB,
                            Result=result)

if __name__=="__main__":
     app.run(debug=True,host="0.0.0.0",port="8080")