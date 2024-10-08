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

def age_of_credit(df:pd.DataFrame)->pd.DataFrame:
    df['earliest_cr_line']=pd.to_datetime(df['earliest_cr_line'])
    df['issue_d']=pd.to_datetime(df['issue_d'])
    df['age_of_credit'] = df['issue_d'].dt.year - df['earliest_cr_line'].dt.year 
    return df[['age_of_credit']]

combined_numerical_pipeline = Pipeline(steps=[
    ('all_numerical',FeatureUnion([
    ('num_skewed_pipeline',num_skewed_pipeline),
    ('num_pipeline',num_pipeline),
    ('age_of_credit_pipeline',FunctionTransformer(age_of_credit))])),
    ('scaling',MinMaxScaler())
])

ordinal_cat_pipeline = Pipeline(steps=[
    ('select_cat_ordinal_features',FunctionTransformer(lambda x:x[config.CAT_ORDINAL_FEATURES])),
    ('categorical_imputer',SimpleImputer(strategy='most_frequent')),
    ('Ordinal_encoder',OrdinalEncoder(categories=config.ORDER_MATRIX,handle_unknown='use_encoded_value',unknown_value=-1))
])

all_nominal_cat = FeatureUnion([
    ('selecting_nominal_features',FunctionTransformer(lambda x:x[config.CAT_NOMINAL_FEATURES].apply(lambda x:x.str.strip().str.lower()))),
    ('home_ownership',FunctionTransformer(lambda x:x['home_ownership'].map({'NONE':'OTHER','ANY':'OTHER'}).fillna(x['home_ownership']).to_frame())),
    ('zipcode_construction',FunctionTransformer(lambda x:x['address'].str.strip().str[-5:].to_frame('zipcode'))),
    ('state_construction',FunctionTransformer(lambda x:x['address'].str.strip().str[-8:-6].to_frame('state')))
])

nominal_cat_pipeline = Pipeline(steps=[
    ('all_nominal_cat',all_nominal_cat),
    ('nominal_imputer',SimpleImputer(strategy='most_frequent')),
    ('Ohe_nominal',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=0.01,sparse_output=False))
])

selected_FE = FeatureUnion([
    ('combined_numerical_pipeline',combined_numerical_pipeline),
    ('nominal_cat_pipeline',nominal_cat_pipeline),
    ('ordinal_cat_pipeline',ordinal_cat_pipeline)
])

target_pipeline = Pipeline(steps=[
    ('target_encoder',FunctionTransformer(lambda x:x.map({'Charged Off':1,'Fully Paid':0}),
                                          inverse_func= lambda x:x.map({0:'Fully Paid',1:'Charged Off'}),
                                          check_inverse=False))
])

selected_FE_with_FS = Pipeline(steps=[
    ('feature_engineering_pipeline',selected_FE),
    ('feature_selection',SelectKBest(k=30,score_func=chi2))
])