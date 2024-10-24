#!/usr/bin/env python
# coding: utf-8

# In[4]:


import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained logistic regression model, scaler, and feature names
with open("C:\\Users\\maheh\\Downloads\\logistic_regression_model.pkl", 'rb') as file:
    model = pickle.load(file)
with open("C:\\Users\\maheh\\Downloads\\scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)
with open("C:\\Users\\maheh\\Downloads\\feature_names.pkl", 'rb') as file:
    feature_names = pickle.load(file)

# Streamlit app header
st.title("Logistic Regression Prediction App")
st.write("This app allows you to predict using a logistic regression model.")

# Collect user input for the features used in training
def get_user_input():
    PassengerId = st.sidebar.text_input('PassengerId', '892')
    Pclass = st.sidebar.selectbox('Pclass', [1, 2, 3])
    Sex = st.sidebar.selectbox('Sex', ['male', 'female'])
    Age = st.sidebar.slider('Age', 0, 100, 30)
    SibSp = st.sidebar.slider('SibSp', 0, 8, 0)
    Parch = st.sidebar.slider('Parch', 0, 8, 0)
    Fare = st.sidebar.slider('Fare', 0.0, 500.0, 50.0)
    Embarked = st.sidebar.selectbox('Embarked', ['C', 'Q', 'S'])
    
    # Convert categorical variables to match the format during training
    Sex = 1 if Sex == 'male' else 0  # Assuming the model was trained with male=1, female=0
    Embarked = {'C': 0, 'Q': 1, 'S': 2}[Embarked]  # Assuming encoding for Embarked
    
    # Create a DataFrame with the input features
    user_data = {
        'PassengerId': int(PassengerId),
        'Pclass': Pclass,
        'Sex': Sex,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
        'Embarked': Embarked
    }
    features = pd.DataFrame(user_data, index=[0])
    return features

# Get user input
input_df = get_user_input()

# Align the user input with the original feature names
input_df_aligned = pd.DataFrame(columns=feature_names)
input_df_aligned = pd.concat([input_df_aligned, input_df], axis=0)
input_df_aligned = input_df_aligned.fillna(0)  # Fill any missing columns with 0s

# Reorder the columns to match the scaler
input_df_aligned = input_df_aligned[feature_names]

# Display the input data
st.subheader('User Input Features:')
st.write(input_df_aligned)

# Scale the input data
input_scaled = scaler.transform(input_df_aligned)

# Make predictions
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# Show the results
st.subheader('Prediction:')
st.write(f'Prediction (0 or 1): {prediction[0]}')
st.write(f'Prediction Probability: {prediction_proba[0]}')

# Display output
if prediction[0] == 1:
    st.success("The model predicts a positive outcome!")
else:
    st.error("The model predicts a negative outcome!")


# In[ ]:




