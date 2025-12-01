import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model

## load the trained model, scaler pickle, onehot

model = load_model('model.keras', compile=False)


## load the encoders and the scaler

with open('label_encode_gender.pkl','rb')as file:
    label_encode_gender = pickle.load(file)

with open('onehot_encode_geo.pkl','rb')as file:
    onehot_encode_geo = pickle.load(file)

with open('scaler.pkl','rb')as file:
    scaler = pickle.load(file)


# streamlit app
st.title('Customer Churn Prediction')

# user input

geography = st.selectbox('Geography', onehot_encode_geo.categories_[0])
gender = st.selectbox('Gender', label_encode_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox(' Is Active Member', [0, 1])

# prepare the input dat

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encode_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# onehot encode geography

geo_encoded = onehot_encode_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encode_geo.get_feature_names_out(['Geography']))

# combine one hot encoded columns with input data 
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# scale the input data
input_scale = scaler.transform(input_data)

# predict churn

prediction = model.predict(input_scale)
prediction_proba = prediction[0][0]
st.write("probability of Customer churning is : ", prediction_proba)


if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')