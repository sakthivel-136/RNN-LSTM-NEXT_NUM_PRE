# streamlit_app.py

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
import os

# Set title
st.title("🔢 RNN Sequence Number Predictor")
st.markdown("Enter a sequence of 3 consecutive numbers to predict the next one.")

# Download model from GitHub if not already
MODEL_URL = "https://github.com/sakthivel-136/RNN-LSTM-NEXT_NUM_PRE/blob/main/rnn_sequence_model.h5"
MODEL_PATH = "rnn_sequence_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        st.success("Model downloaded successfully!")

# Load the model
model = load_model(MODEL_PATH)

# Input fields
col1, col2, col3 = st.columns(3)
with col1:
    n1 = st.number_input("Number 1", value=1)
with col2:
    n2 = st.number_input("Number 2", value=2)
with col3:
    n3 = st.number_input("Number 3", value=3)

# Predict button
if st.button("Predict Next Number"):
    # Prepare input
    input_seq = np.array([n1, n2, n3]).reshape((1, 3, 1))

    # Prediction
    predicted = model.predict(input_seq, verbose=0)
    st.success(f"📈 Predicted Next Number: **{predicted[0][0]:.2f}**")
