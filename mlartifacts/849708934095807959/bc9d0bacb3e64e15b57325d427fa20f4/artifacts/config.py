import os

PARENT_ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
FILE_NAME = 'LoanTap_data.csv'
URL=''

# Keeping all the variables which will be used inside the pipeline in capital letters
# Training configuration
N_JOBS = -1
RANDOM_SEED = 42 

CAT_ORDINAL_FEATURES = ['term','grade','sub_grade','emp_length','verification_status']

# We need to specify order for the ordinal features so making some lists for that
term_order = [' 36 months', ' 60 months']
grade_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
sub_grade_order = [grade + str(i) for grade in grade_order for i in range(1,6)]
emp_length_order = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
verification_status_order = ['Not Verified', 'Verified', 'Source Verified']

ORDER_MATRIX = [term_order, grade_order, sub_grade_order, emp_length_order, verification_status_order]

CAT_NOMINAL_FEATURES = ['home_ownership','purpose','title','initial_list_status','application_type']

NUM_FEATURES = ['loan_amnt','revol_util']

NUM_SKEWED_FEATURES = ['annual_inc','dti','open_acc', 'pub_rec','revol_bal', 'total_acc', 'mort_acc','pub_rec_bankruptcies']

TARGET = 'loan_status'