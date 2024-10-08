import logging
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder, FunctionTransformer, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest,chi2
import os
import sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import credit_risk_model.config as config 
print(config.TARGET)

# Num skewed pipeline
num_skewed_pipeline = Pipeline(steps=[
    ('select_skewed_features',FunctionTransformer(lambda x:x[config.NUM_SKEWED_FEATURES])),
    ('skewed_imputer',SimpleImputer(strategy='median')),
    ('log_transform',FunctionTransformer(lambda x: np.log(x+1e-5)))
])

num_pipeline = Pipeline(steps=[
    ('select_numerical_features',FunctionTransformer(lambda x:x[config.NUM_FEATURES])),
    ('numerical_imputer',SimpleImputer(strategy='mean'))
])