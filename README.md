# Credit Risk Model

## Project Overview
With ample financial data available, the Banking Institute aims to understand key determinants in extending credit lines. This analysis assists the Bank in grasping significant loan approval factors and their interplay. The ultimate goal is to devise a model predicting if a credit line should be extended based on individual attributes.  The model is trained on a dataset of borrower attributes, including financial, demographic, and credit-related features. The goal is to enable data-driven decisions in the loan underwriting process.

### Project Breakdown

* Predictive modeling using machine learning algorithms.
* Feature engineering and selection with sklearn pipelines.
* Hyperparameter tuning using Optuna.
* MLflow for experiment tracking.
* UI built using streamlit, Flask.
* API end point created using FASTAPI.
* Deployed using Docker.

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

## Model

The model used for this project is an XGBoost classifier.

## Results

The model achieved an f1 score of 65% on the test set.

## License

This project is licensed under the MIT License.
