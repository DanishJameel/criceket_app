import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('cricket_model.joblib')
    return model

def predict_output(model, input_data):
    prediction = model.predict(input_data)
    return prediction

def main():
    model = load_model()

    st.title('Cricket Prediction App')
    st.write('Enter the following details to make predictions:')

    opposition_dict = {
        'England': 0,
        'Pakistan': 1,
        'Sri Lanka': 2,
        'South Africa': 3,
        'Australia': 4,
        'India': 5,
        'New Zealand': 6,
        'West Indies': 7,
        'Afghanistan': 8,
        'Bangladesh': 9
    }

    opposition = st.selectbox('Opposition', list(opposition_dict.keys()))
    team_batters = st.number_input('Number of Team Batters')
    team_allrounders = st.number_input('Number of Team Allrounders')
    team_bowlers = st.number_input('Number of Team Bowlers')
    team_rh_bat = st.number_input('Number of Team RH Bat')
    team_lh_bat = st.number_input('Number of Team LH Bat')
    team_fast = st.number_input('Number of Team Fast')
    team_spin = st.number_input('Number of Team Spin')

    input_data = np.array([
        opposition_dict[opposition],
        team_batters,
        team_allrounders,
        team_bowlers,
        team_rh_bat,
        team_lh_bat,
        team_fast,
        team_spin
    ]).reshape(1, 8)

    if st.button('Predict'):
        prediction = predict_output(model, input_data)
        st.write('Predicted output for the 7 target variables:', prediction)

if __name__ == '__main__':
    main()
