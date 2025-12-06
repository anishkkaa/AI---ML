import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from joblib import load

st.set_page_config(
    page_title="Blood Donation Predictor",
    page_icon="🩸",
    layout="wide"
)

#load the model
@st.cache_resource
def load_model():
    return load('blood_donation_model.joblib')

model = load_model()
#sidebaar
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ["Prediction", "About"])

if page == "Prediction":
    #Main Content
    st.title("Blood Donation Prediction App ")
    st.write("----")

 #create two columns   
    col1 , col2 = st.columns(2)
    
    with col1:
        st.subheader("Enter Donor Informatio")
        recency = st.slider("Months since last donation", 0, 100, 5)
        frequency = st.slider("Total number of donations", 0, 50, 1)
        monetary = st.slider("Total blood donated (in c.c.)", 0, 5000, 250)
        time = st.slider("Months since first donation", 0, 200, 10)
    with col2: 
        st.subheader("Current Values")
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Recency", f"{recency} months")
            st.metric("Frequency", f"{frequency} times")
        with metrics_col2:   
            st.metric("Blood Volume", f"{monetary} c.c.")
            st.metric("Time", f"{time} months")

#prediction logic

st.write("----")
if st.button("Predict Donation", type="primary"):
    with st.spinner("Processing..."):
        input_data = pd.DataFrame({
            'Recency': [recency],
            'Frequency': [frequency],
            'Monetary': [monetary],
            'Time': [time]
        })

        prediction = model.predict(input_data)
        #probability = model.predict_proba(input_data)


        #Display results
        #col1, col2 = st.columns(3)

    
        if prediction[0] == 1:
                st.success("Likely to Donate !!")
        else:
                st.error("Unlikely to Donate !!")    
    