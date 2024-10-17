import streamlit as st
import pandas as pd
from credit_risk_model import data_processor

model = data_processor.load_pipeline('XGB_model')

def main():
    st.markdown("## Welcome to Loan status predictor🏦")
    st.markdown("##### Please enter your details to check whether loan will be approved or not")
    input_fields = {
        'loan_amnt': st.number_input("Loan Amount", min_value=0.0),
        'term': st.radio("Term", (" 36 months", " 60 months")),
        'int_rate': st.number_input("Interest Rate", min_value=0.0, max_value=100.0),
        'installment': st.number_input("Installment", min_value=0.0),
        'grade': st.selectbox("Grade", ("A", "B", "C", "D", "E", "F", "G")),
        'sub_grade': st.selectbox("Sub Grade", [f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)]),
        'emp_title': st.text_input("Employment Title"),
        'emp_length': st.selectbox("Employment Length", ("< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years")),
        'home_ownership': st.selectbox("Home Ownership", ("RENT", "OWN", "MORTGAGE", "OTHER")),
        'annual_inc': st.number_input("Annual Income", min_value=0.0),
        'verification_status': st.selectbox("Verification Status", ("Verified", "Not Verified", "Source Verified")),
        'issue_d': st.date_input("Issue Date"),
        'purpose': st.radio("Purpose", ("debt_consolidation", "credit_card", "home_improvement", "other")),
        'title': st.text_input("Loan Title"),
        'dti': st.number_input("Debt-to-Income Ratio", min_value=0.0),
        'earliest_cr_line': st.date_input("Earliest Credit Line"),
        'open_acc': st.number_input("Number of Open Credit Lines", min_value=0),
        'pub_rec': st.number_input("Number of Derogatory Public Records", min_value=0),
        'revol_bal': st.number_input("Revolving Balance", min_value=0.0),
        'revol_util': st.number_input("Revolving Utilization Rate", min_value=0.0, max_value=100.0),
        'total_acc': st.number_input("Total Number of Credit Lines", min_value=0),
        'initial_list_status': st.selectbox("Initial List Status", ("w", "f")),
        'application_type': st.selectbox("Application Type", ("INDIVIDUAL", "JOINT")),
        'mort_acc': st.number_input("Number of Mortgage Accounts", min_value=0),
        'pub_rec_bankruptcies': st.number_input("Number of Public Record Bankruptcies", min_value=0),
        'address': st.text_input("Address")
    }
    st.info("This app is trained on data from LoanTap Financial Technologies")
    if st.button("Predict"):
        # Convert date fields to string format
        input_fields['issue_d'] = input_fields['issue_d'].strftime("%b-%Y")
        input_fields['earliest_cr_line'] = input_fields['earliest_cr_line'].strftime("%b-%Y")

        result = model.predict(pd.DataFrame([input_fields]))[0]
        if result == 0:
            st.success("Your loan Application is Approved")
        else:
            st.error("Your loan Application is Rejected")

    st.sidebar.header("Loan status prediction")
    st.sidebar.text("created by Sreekar using streamlit")









if __name__=="__main__":
    main()