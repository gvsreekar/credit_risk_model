import sys,os
import numpy as np 
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import credit_risk_model.config as config
from sklearn.model_selection import TunedThresholdClassifierCV
from credit_risk_model.FE_pipeline import target_pipeline
from sklearn.metrics import classification_report
from credit_risk_model.plotting import plot_threshold_scoring

def find_the_best_decision_threshold(model,x_train,y_train,x_test,y_test,scoring='f1')->pd.DataFrame:
    best_threshold_model = TunedThresholdClassifierCV(model,scoring='f1',cv=3,n_jobs=config.N_JOBS,store_cv_results=True)
    y_train_transformed = target_pipeline.transform(y_train)
    y_test_transformed = target_pipeline.transform(y_test)
    best_threshold_model.fit(x_train,y_train_transformed)
    y_pred = best_threshold_model.predict(x_test)
    print('Classification report: Training set')
    print(classification_report(y_train_transformed,best_threshold_model.predict(x_train)))
    print('Classification report: Test set')
    print(classification_report(y_test_transformed,y_pred))
    print(f'Best threshold = {best_threshold_model.best_threshold_:.2f} with {scoring} score = {best_threshold_model.best_score_:.2f}')
    plot_threshold_scoring(best_threshold_model,scoring)
    return best_threshold_model