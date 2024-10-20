from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import uvicorn
import pandas as pd
from credit_risk_model import data_processor

app = FastAPI()

class LoanFeatures(BaseModel):
    loan_amnt: float
    term: str
    int_rate: float
    installment: float
    grade: str
    sub_grade: str
    emp_title: str
    emp_length: str
    home_ownership: str
    annual_inc: float
    verification_status: str
    issue_d: str
    purpose: str
    title: str
    dti: float
    earliest_cr_line: str
    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: float
    total_acc: float
    initial_list_status: str
    application_type: str
    mort_acc: float
    pub_rec_bankruptcies: float
    address: str

model = data_processor.load_pipeline('XGB_model')

@app.get('/')
def index():
    return {'message': 'Welcome to Loan Prediction App'}

@app.post('/predict')
async def predict(loan_data: LoanFeatures):
    """
    Predicts whether a loan should be approved or not based on the input data.
    """
    # Extract the data from the input
    data = loan_data.model_dump()
    df = pd.DataFrame([data])
    pred = model.predict(df)
    if pred[0] == 0:
        return {'Status of Loan Application': 'Approved'}
    else:
        return {'Status of Loan Application': 'Rejected'}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)