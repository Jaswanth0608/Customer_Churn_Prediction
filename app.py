# import numpy as np
# from flask import Flask, request, jsonify, render_template #render_template help in getting back to home page.
# import pickle

# app = Flask(__name__)
# model = pickle.load(open("Customer_churn_model.pkl","rb"))

# @app.route("/") #need to be used create any number of uri's
# def home():
#      return render_template("index.html")

# def pred(pred_list):
#      preds = np.array(pred_list).reshape(1,6)
#      loded_model = pickle.load(open("Customer_churn_model.pkl",'+rb'))
#      result = loded_model.predict(preds)
#      return result[0]

# @app.route("/predict", methods=['POST'])
# def predict():
#      int_features = [float(x) for x in request.form.values()]
#      # Age = request.form['Age']
#      # Gender = request.form['Gender']
#      # Location = request.form['Location']
#      # Subscription_Length_Months = request.form['Subscription_Length_Months']
#      # Monthly_Bill = request.form['Monthly_Bill']
#      # Total_Usage_GB = request.form['Total_Usage_GB']
#      features = [np.array(int_features)]
#      prediction = model.predict(features)
#      output = round(prediction[0],2)

#      return render_template('home.html',prediction_text = 'The Customer will {}'.format(output))

# if __name__ == "main":
#      app.run()



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
          # result = _predict(Age,Gender,Location,Subscription_Length_Months,Monthly_Bill,Total_Usage_GB)
          # print(Gender,Location)
          # Gender_ = ""
          # Location_ = ""
          # Gender_ = genders_[Gender]
          # Location_ = locations_[Location]   
          # print(Gender_,Location_)

          data_dict = {
          'Age': [Age],
          'Gender': [Gender],
          'Location': [Location],
          'Subscription_Length_Months': [Subscription_Length_Months],
          'Monthly_Bill': [Monthly_Bill],
          'Total_Usage_GB': [Total_Usage_GB]
          }

          print(data_dict)

          # Convert the dictionary to a DataFrame
          df = pd.DataFrame(data_dict)
          print (df)
          df[['Location','Gender']] = df[['Location','Gender']].apply(lambda x: label_encoder.fit_transform(x))
          print(df)
          df = minmaxscaler.transform(df)
          print(df)
          pred = model.predict(df)
          print(pred,'\n',pred[0])
          result = pred[0]

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