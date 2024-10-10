import sys,os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from credit_risk_model.FE_pipeline import selected_FE_with_FS,target_pipeline
from xgboost import XGBClassifier
from credit_risk_model import config
from credit_risk_model.data_processor import load_pipeline,save_pipeline,load_data_and_sanitize,save_data
import logging
from credit_risk_model.tune_threshold import find_the_best_decision_threshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,train_test_split

XGB_pipe = Pipeline(steps=[
    ('feature_engineering_pipeline',selected_FE_with_FS),
    ('xgb',XGBClassifier())
])
params = {'xgb__learning_rate': [0.2],
 'xgb__max_depth': [5],
 'xgb__min_samples_leaf': [12],
 'xgb__n_estimators': [250]}

SCORING = 'f1'

gridxgb = GridSearchCV(XGB_pipe,param_grid=params,scoring='f1',cv=3,verbose=1,n_jobs=config.N_JOBS)

def perform_training():
    df = load_data_and_sanitize(config.FILE_NAME)
    x = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,stratify=y,test_size=0.3)
    # Let's add df test to CSV and store it in a separate test file
    x_test[config.TARGET]=y_test
    save_data(x_test,'test_data.csv')
    
    y_train_transformed = target_pipeline.fit_transform(y_train)
    XGB_best_model = gridxgb.fit(x_train,y_train_transformed).best_estimator_
    XGB_tuned_model = find_the_best_decision_threshold(XGB_best_model,x_train,y_train,x_test,y_test,scoring=SCORING)
    
    save_pipeline(XGB_tuned_model,'XGB_model')
    save_pipeline(target_pipeline,'target_pipeline')
    
if __name__=='__main__':
    perform_training()
    
    
