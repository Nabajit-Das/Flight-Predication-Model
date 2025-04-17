import streamlit as st
import pandas as pd
import pickle
import datetime

# Load trained model
with open("flight_rf.pkl", "rb") as file:
    model = pickle.load(file)

# Load expected model columns
with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

st.title("✈️ Flight Price Prediction App")

# Sidebar inputs
st.sidebar.header("Enter Flight Details")

# Date of Journey
journey_date = st.sidebar.date_input("Date of Journey", datetime.date.today())

# Departure Time
dep_time = st.sidebar.time_input("Departure Time", datetime.time(10, 0))

# Arrival Time
arrival_time = st.sidebar.time_input("Arrival Time", datetime.time(12, 0))

# Compute Duration
def compute_duration(dep, arr):
    dep_dt = datetime.datetime.combine(datetime.date.today(), dep)
    arr_dt = datetime.datetime.combine(datetime.date.today(), arr)
    if arr_dt < dep_dt:
        arr_dt += datetime.timedelta(days=1)
    duration = arr_dt - dep_dt
    return duration.seconds // 3600, (duration.seconds // 60) % 60

# Stops
stops = st.sidebar.selectbox("Total Stops", ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"])

# Airline
airline = st.sidebar.selectbox("Airline", [
    "IndiGo", "Air India", "Jet Airways", "SpiceJet", "Vistara",
    "GoAir", "Multiple carriers", "Air Asia", "Trujet",
    "Multiple carriers Premium economy", "Jet Airways Business", "Vistara Premium economy"
])

# Source
source = st.sidebar.selectbox("Source", ["Delhi", "Kolkata", "Mumbai", "Chennai", "Banglore"])

# Destination
destination = st.sidebar.selectbox("Destination", ["Cochin", "Delhi", "New Delhi", "Hyderabad", "Kolkata", "Banglore"])

# Additional Info
info = st.sidebar.selectbox("Additional Info", [
    "No info", "In-flight meal not included", "No check-in baggage included", "1 Short layover",
    "1 Long layover", "Change airports", "Business class", "Red-eye flight", "2 Long layover"
])

# Predict button
if st.sidebar.button("Predict Price"):
    # Feature engineering
    journey_day = journey_date.day
    journey_month = journey_date.month
    dep_hour = dep_time.hour
    dep_min = dep_time.minute
    arr_hour = arrival_time.hour
    arr_min = arrival_time.minute
    dur_hour, dur_min = compute_duration(dep_time, arrival_time)

    # Initialize input DataFrame with model columns
    input_data = pd.DataFrame(data=[0]*len(model_columns), index=model_columns).T

    # Fill known features
    input_data.at[0, 'Journey_day'] = journey_day
    input_data.at[0, 'Journey_month'] = journey_month
    input_data.at[0, 'Dep_hour'] = dep_hour
    input_data.at[0, 'Dep_min'] = dep_min
    input_data.at[0, 'Arrival_hour'] = arr_hour
    input_data.at[0, 'Arrival_min'] = arr_min
    input_data.at[0, 'Duration_hours'] = dur_hour
    input_data.at[0, 'Duration_mins'] = dur_min

    # One-hot encoded fields (set only if present)
    fields_to_encode = [
        f'Total_Stops_{stops}',
        f'Airline_{airline}',
        f'Source_{source}',
        f'Destination_{destination}',
        f'Additional_Info_{info}'
    ]
    for col in fields_to_encode:
        if col in input_data.columns:
            input_data.at[0, col] = 1

    # Predict
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Flight Price: ₹ {round(prediction, 2)}")
