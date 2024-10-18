from flask import Flask,render_template,request
import pandas as pd

from credit_risk_model import data_processor

model = data_processor.load_pipeline('XGB_model')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles POST requests from the homepage form, makes a prediction and renders the homepage with the result.
    
    :return: Rendered homepage with the result of the prediction
    """
    if request.method == 'POST':
        request_data = dict(request.form)
        data = pd.DataFrame([request_data])
        print(data)
        pred = model.predict(data)
        print(f"prediction is {pred}")

        if int(pred[0]) == 0:
            result = "Congratulations! Your loan application is approved"
        else:
            result = "Sorry! Your loan application is rejected"
        return render_template('home.html', prediction = result)

@app.errorhandler(500)
def internal_error(error):
    return "500: Something went wrong"

@app.errorhandler(404)
def not_found(error):
    return "404: Page not found", 404

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080) # By default it will run on 5000, since i have mlflow running on 
    # 5000, I will run this in port 8080