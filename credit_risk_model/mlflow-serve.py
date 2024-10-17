import mlflow
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from credit_risk_model import data_processor
from credit_risk_model import config

# Set the MLflow tracking URI if not already set
mlflow.set_tracking_uri("http://localhost:5000")

# Define the model URI (model name and version/stage in MLflow Model Registry)
model_uri = "models:/Optuna optimized XGB classifier@challenger"  # Use @latest for latest version or specific version like @1


loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)
import pandas as pd 

sample = data_processor.load_data_and_sanitize(config.FILE_NAME)

print(f"Prediction is {loaded_model.predict(sample.iloc[[0]])}")