from flask import Flask, request, render_template 
import numpy as np
from joblib import load
import pandas as pd
   
app = Flask(__name__) 
  
@app.route('/', methods=['GET']) 
def index(): 
    return render_template('index.html') 
  
@app.route('/read-form', methods=['POST']) 
def read_form(): 
    data = request.form

    userArray = {
        "Age": [data["userAge"]],
        "Gender": [data["userGender"]],
        "Occupation": [data["userOccupation"]],
        "Travel Class": [data["userTarvelClass"]],
        "Destination": [data["userDestination"]],
        "Distance to Destination (Light-Years)": [data["userDistance"]],
        "Duration of Stay (Earth Days)": [data["userDuration"]],
        "Number of Companions": [data["userCompanion"]],
        "Purpose of Travel": [data["userPurpose"]],
        "Transportation Type": [data["userTransportation"]],
        "Special Requests": [data["userRequests"]],
        "Loyalty Program Member": [data["userLoyalty"]],
        "Month": [data["userMonth"]]
    }

    df = pd.DataFrame(userArray)
    
    loaded_model = load(open("Dumps\est.pkl", 'rb'))

    predict = loaded_model.predict(df)

    print(predict)

    answer = str(predict[0])

    print(answer)

    return answer
  
if __name__ == '__main__': 
    app.run()