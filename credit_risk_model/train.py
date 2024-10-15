import sys,os
import numpy as np
import pandas as pd
import mlflow
import optuna
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from credit_risk_model.FE_pipeline import selected_FE_with_FS,target_pipeline
from xgboost import XGBClassifier
from credit_risk_model import config
from credit_risk_model import FE_pipeline
from credit_risk_model.data_processor import load_pipeline,save_pipeline,load_data_and_sanitize,save_data
import logging
from credit_risk_model.tune_threshold import find_the_best_decision_threshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import f1_score
# mlflow.set_tracking_uri('http://127.0.0.1:5000')


df = load_data_and_sanitize(config.FILE_NAME)
x = df.drop(columns=[config.TARGET])
y = df[config.TARGET]
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,stratify=y,test_size=0.3)
# Let's add df test to CSV and store it in a separate test file
x_test[config.TARGET]=y_test
save_data(x_test,'test_data.csv')

# Transform the target
y_train_transformed = target_pipeline.fit_transform(y_train)
# Transform test to match the target pipeline
y_test_transformed = target_pipeline.transform(y_test)
SCORING = 'f1'

def objective(trial):
    logging.info('Starting objective function for optuna trial')
    
    # Start mlflow run to log hyperparameters and also metrics for each run of Optuna
    # When you use MLflow's start_run() within a with statement, you don't need to manually end the run.
    # The with keyword ensures that the run is automatically closed (or ended) when the block is exited
    with mlflow.start_run(nested=True,run_name=f"Trial_{trial.number+1}"):
        param_grid = {
            'n_estimators':trial.suggest_int('n_estimators',100,250),
            'max_depth': trial.suggest_int('max_depth',2,8),
            'scale_pos_weight' : trial.suggest_float('scale_pos_weight',1,4),
            # When log=True, the values for learning_rate are sampled exponentially, meaning values closer 
            # to 1e-3 (lower bound) have a higher probability of being selected compared to 
            # values closer to 0.2 (upper bound).
            'learning_rate':trial.suggest_float('learning_rate',1e-3,0.2,log=True),
            # By setting 'eval_metric': 'aucpr' in XGBoost, you tell the model to use AUCPR to evaluate its 
            # performance on the validation set at each boosting iteration. The metric helps guide the 
            # model's training by focusing on the quality of positive class predictions, 
            # particularly useful for imbalanced datasets.
            # default eval metric for xgboost binary classification is logloss
            'eval_metric':'aucpr',
            'subsample': trial.suggest_float('subsample', 0.6, 1.0)
        }
        XGB_model = Pipeline(steps=[
            ('feature_engineering_pipeline',selected_FE_with_FS),
            ('xgb',XGBClassifier(**param_grid))
        ])
        # Let's log hyperparameters in mlflow
        mlflow.log_params(param_grid)
        XGB_model.fit(x_train,y_train_transformed)
        y_pred = XGB_model.predict(x_test)
        # Storing f1 of the target class i.e 1
        f1_class_1 = f1_score(y_test_transformed,y_pred,pos_label=1)
        mlflow.log_metric('f1_score',f1_class_1)
        
    return f1_class_1
        
    
    

def perform_training():
    
    # we have to create a study and maximize it using optuna 
    mlflow.set_experiment("Optuna optimization of XGB classifier")
    with mlflow.start_run(run_name="Optuna optimized model after 30 trials"):
        mlflow.set_tag('model','XGBClassifier')
        mlflow.set_tag('objective','maximize_f1_class_1')
        
        study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler())
        study.optimize(objective,n_trials=30,show_progress_bar=True)
        best_params = study.best_trial.params
        
        XGB_model = Pipeline(steps=[
            ('feature_engineering_pipeline',selected_FE_with_FS),
            ('xgb',XGBClassifier(**best_params))
        ])
        mlflow.log_params(best_params)
        fitted_xgb_model = XGB_model.fit(x_train,y_train_transformed)
        XGB_tuned_model,report = find_the_best_decision_threshold(fitted_xgb_model,x_train,y_train,x_test,y_test,scoring=SCORING)
        # report['1'] will capture f1 score, precision and recall of 
        mlflow.log_metrics(report['1'])
        mlflow.log_metric('threshold',XGB_tuned_model.best_threshold_)
        # Use the below code to log artifact if you run the code from root directory using mlflow run
        mlflow.log_artifact('credit_risk_model/FE_pipeline.py')
        mlflow.log_artifact('credit_risk_model/config.py')
        # Use the below code if you run only train.py file by being in the train.py file path
        # mlflow.log_artifact('FE_pipeline.py')
        # mlflow.log_artifact('config.py')
        mlflow.sklearn.log_model(XGB_tuned_model, 'model')
    
    save_pipeline(XGB_tuned_model,'XGB_model')
    save_pipeline(target_pipeline,'target_pipeline')
    
if __name__=='__main__':
    perform_training()
    
    
