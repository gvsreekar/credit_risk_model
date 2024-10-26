# End-to-End Deployable Credit Risk Model

## [LIVE DEMO](https://creditriskmodel-aprewf5gesj6uqmo5y3ygx.streamlit.app/) -> [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://creditriskmodel-aprewf5gesj6uqmo5y3ygx.streamlit.app/)

## Project Overview
This project aims to develop an End-to-End machine learning model to predict loan default risk for a financial services platform.
With ample financial data available, the Banking Institute aims to understand key determinants in extending credit lines. This analysis assists the Bank in grasping significant loan approval factors and their interplay. The model is trained on a dataset of borrower attributes, including financial, demographic, and credit-related features. The goal is to enable data-driven decisions in the loan underwriting process.

### Tech used

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)![Pandas](https://img.shields.io/badge/Pandas-1.x-brightgreen?logo=pandas)![NumPy](https://img.shields.io/badge/NumPy-1.x-orange?logo=numpy)![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-blueviolet?logo=plotly)![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24%2B-lightgrey?logo=scikit-learn)![MLFlow](https://img.shields.io/badge/MLFlow-1.x-blue?logo=mlflow)![Optuna](https://img.shields.io/badge/Optuna-3.x-red?logo=optuna)![FastAPI](https://img.shields.io/badge/FastAPI-0.85%2B-brightgreen?logo=fastapi)![Streamlit](https://img.shields.io/badge/Streamlit-1.x-orange?logo=streamlit)

### Project Breakdown

* Predictive modeling using machine learning algorithms.
* Feature engineering and selection with sklearn pipelines.
* Hyperparameter tuning using Optuna.
* MLflow for experiment tracking and model registry.
* Frontend with streamlit App
* API end point created using FASTAPI.
* Containerization using Docker.

## Installation/Environment Setup

### Prerequisites

* Python (>=3.7)
* Poetry (>=1.8)
* Dependencies listed in `pyproject.toml`

### Installation Instructions

1. Clone the repository.

2. Download dependecies using:
```bash
poetry install
```

3. Activate the poetry environment using the below command:
```bash
poetry shell
```
## Project Structure
The following is the structure of the project to help you navigate through it:
```ASCI
Credit_Risk_Model/
┣ data/ - Contains raw data files for the project
┃ ┣ loantap_data.csv - Main dataset for training and testing
┃ ┗ test_data.csv - Test dataset for evaluating model performance (sampled from above dataset)
┣ credit_risk_model/ - Contains source code for the project
┃ ┣ __init__.py - Initialization file for the package
┃ ┣ trained_models/ - Directory to store trained machine learning models
┃ ┃ ┣ XGB_model.pkl - Main trained model
┃ ┃ ┗ target_pipeline.pkl - Fitted target pipeline for finding inverse after prediction
┃ ┣ FE_pipeline.py - Feature engineering pipeline configurations
┃ ┣ config.py - Configuration file for project settings
┃ ┣ data_processor.py - Script for data loading, cleaning, and saving data and pipelines
┃ ┣ tune_threshold.py - Script for choosing the decision threshold
┃ ┣ plotting.py - Script for creating visualizations
┃ ┣ predict.py - Script for making predictions with trained models
┃ ┗ train.py - Script for training machine learning models
┣ notebooks/ - Directory for Jupyter notebooks
┃ ┗ model_prototyping.ipynb - Notebook for EDA, prototyping and testing models
┣ tests/ - Directory for unit tests and integration tests
┃ ┗ data_test.py - Script for testing data handling functions
┃.gitignore - File specifying files to ignore in Git version control
┣ README.md - Project README file
┣ MLProject - Text file to run the best model using mlflow command.
┣ flask_app.py - Script for flask app.
┣ conda.yaml - yaml file with dependencies to create mlflow environment.
┣ model_building.ipynb - Inital notebook for analysis.
┣ fastapi_app.py - Script for creating the API for the model.
┣ Dockerfile - Docker file for dockerising the fastapi endpoint.
┗ pyproject.toml - Project project dependencies file using Poetry

```
## Usage

Make sure to add poetry to the path. If not added, then call poetry with the complete path to avoid getting error saying poetry not found.

To train the model, run the following command in the root directory:

```bash
poetry run python credit_risk_model/train.py
```

To make predictions, run the following command in the root directory:

```bash
poetry python credit_risk_model/predict.py
```
### Running Web Apps

#### fastAPI

   ```bash
   poetry run python fastapi_app.py 
   ```

   POST to `localhost:8000/predict` with Postman or use `localhost:8000/predict/docs` in browser for documentation / testing


## Usage with docker

### 1. Pulling the Docker Image

To pull the Docker image from Docker Hub, run the following command:

```sh
# Pull the docker image
docker pull gvsreekar/loantap_api:v2
```

### 2. Running the Docker Container

To run the Docker container, use the following command:

```sh
# Run the docker container
docker run -p 8000:8000 loantap_api:v2 # goto http://localhost:8501 in browser
```

## Model

The model used for this project is an XGBoost classifier.

## Results

The model achieved an f1 score of **66.12%** and recall of **85.84%** on the test dataset.

## License

This project is licensed under the MIT License.
