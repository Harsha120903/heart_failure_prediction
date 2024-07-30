#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as no
import pickle


# In[6]:


with open('rf_model_heart.pkl','rb') as file:
    model=pickle.load(file)
st.title('Heart failure prediction App')
st.sidebar.header('Patient Data')
def user_input_features():
    age = st.sidebar.number_input('Age', min_value=0, key='age')
    anaemia = st.sidebar.selectbox('Anaemia', [0, 1], key='anaemia')
    creatinine_phosphokinase = st.sidebar.number_input('Creatinine Phosphokinase', min_value=0, key='cp')
    diabetes = st.sidebar.selectbox('Diabetes', [0, 1], key='diabetes')
    ejection_fraction = st.sidebar.number_input('Ejection Fraction', min_value=0, key='ef')
    high_blood_pressure = st.sidebar.selectbox('High Blood Pressure', [0, 1], key='hbp')
    platelets = st.sidebar.number_input('Platelets', min_value=0.0, key='platelets')
    serum_creatinine = st.sidebar.number_input('Serum Creatinine', min_value=0, key='sc')
    serum_sodium = st.sidebar.number_input('Serum Sodium', min_value=0, key='ss')
    sex = st.sidebar.selectbox('Sex', [0, 1], key='sex')
    smoking = st.sidebar.selectbox('Smoking', [0, 1], key='smoking')
    time = st.sidebar.number_input('Time', min_value=0, key='time')
    
    data = {
        'age': age,
        'anaemia': anaemia,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes': diabetes,
        'ejection_fraction': ejection_fraction,
        'high_blood_pressure': high_blood_pressure,
        'platelets': platelets,
        'serum_creatinine': serum_creatinine,
        'serum_sodium': serum_sodium,
        'sex': sex,
        'smoking': smoking,
        'time': time  # Note: Removed the extra space before 'time'
    }
    features = pd.DataFrame(data, index=[0])
    return features

df=user_input_features()
st.subheader('Patient Data')
st.write(df)

if st.button('Predict'):
    prediction=model.predict(df)
    prediction_proba=model.predict_proba(df)

    st.subheader('Prediction')
    heart_failure_risk=np.array(['No','Yes'])
    st.write(f'Heart Failure:{heart_failure_risk[prediction][0]}')

    st.subheader('Prediction Probability')
    st.write(prediction_proba)


# In[ ]:





# In[ ]:




