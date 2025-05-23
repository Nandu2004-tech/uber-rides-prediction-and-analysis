import streamlit as st
import pandas as pd
import numpy as np
import pickle
from math import radians, sin, cos, sqrt, atan2

st.title('Uber Fare Prediction')

# Function to load the trained model
@st.cache_resource
def load_model():
    with open('svr_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to calculate distance between coordinates using Haversine formula
def calculate_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return 6371 * c  # Earth radius in kilometers

# Load the model
model = load_model()

# User input
st.subheader('Enter the Pickup and Dropoff Details')
pickup_longitude = st.number_input('Pickup Longitude', value=0.0)
pickup_latitude = st.number_input('Pickup Latitude', value=0.0)
dropoff_longitude = st.number_input('Dropoff Longitude', value=0.0)
dropoff_latitude = st.number_input('Dropoff Latitude', value=0.0)
passenger_count = st.number_input('Passenger Count', min_value=1, max_value=6, value=1)

# Compute distance
distance_km = calculate_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)

# Predict fare
if st.button('Predict Fare'):
    input_data = pd.DataFrame({
        'pickup_longitude': [pickup_longitude],
        'pickup_latitude': [pickup_latitude],
        'dropoff_longitude': [dropoff_longitude],
        'dropoff_latitude': [dropoff_latitude],
        'passenger_count': [passenger_count],
        'distance_km': [distance_km]
    })

    prediction = model.predict(input_data)
    st.subheader('Predicted Fare Amount')
    st.write(f'${prediction[0]:.2f}')
