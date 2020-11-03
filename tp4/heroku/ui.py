import streamlit as st
import pandas as pd
import joblib

user_input = st.text_input("Enter english text:", "")
model = joblib.load("model.joblib")
if user_input:
    proba = model.predict_proba([user_input])[0]    
    if proba[0] > proba[1] and proba[0] > proba[2]:
    	st.text("hate_speech")
    elif proba[1] > proba[0] and proba[1] > proba[2]:
        st.text("offensive_language")
    else:
        st.text("neither")