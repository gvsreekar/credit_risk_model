import logging
import sys,os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from credit_risk_model.data_processor import load_pipeline,load_data_and_sanitize
from credit_risk_model import config
from sklearn.metrics import classification_report
import pandas as pd

model = load_pipeline('XGB_model')
target_pipeline = load_pipeline('target_pipeline')

def generate_prediction():
    logging.info('Starting prediction')
    test_data = load_data_and_sanitize('test_data.csv')
    y = test_data[config.TARGET]
    x = test_data.drop(config.TARGET,errors='ignore')
    y_pred_transformed = model.predict(x)
    logging.info('Finished prediction')
    y_pred_transformed = pd.Series(y_pred_transformed)
    y_pred = target_pipeline.inverse_transform(y_pred_transformed)
    print(classification_report(y, y_pred))
    return y_pred

if __name__=='__main__':
    generate_prediction()