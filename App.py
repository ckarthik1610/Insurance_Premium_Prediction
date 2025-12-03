import streamlit as st
import pandas as pd
import joblib

import Car_Insurance
import Health_Insurance
import Home_Insurance

st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Insurance Premium Predictor")

st.markdown(
    """
    Select an insurance type below, enter the details, and get an estimated premium.
    """
)

pills = st.pills("Choose Insurance Type",["Car Insurance", "Home Insurance", "Health Insurance"], default = "Car Insurance", label_visibility="collapsed")

if pills == "Car Insurance":
    st.header("🚗 Car Insurance – Premium Prediction")

    with st.form("car_form"):
        age = st.number_input("Driver Age",min_value = 18,max_value = 65,value = 25)
        annual_km = st.number_input("Annual Mileage(km)",min_value = 10000,max_value = 25000,value = 15000)
        car_age = st.number_input("Car Age",min_value=0,max_value=35,value = 5)
        exp_years = st.number_input("Driver Experience(Years)",min_value=0,max_value=40,value = 5)
        num_accidents = st.number_input("No of Accidents",min_value = 0,max_value = 5,value = 0)
        vehicle_type = st.selectbox("Vehicle Type",["sedan","suv","sports","truck"])

        submitted = st.form_submit_button("Predict Car Premium")

        if submitted:

            premium = Car_Insurance.training.result(age=age,annual_km=annual_km,car_age=car_age,exp_years=exp_years,num_accidents=num_accidents,vehicle_type=vehicle_type,)

            st.success(f"Estimated Car Insurance Premium: **${premium:,.2f}**")

elif pills == "Home Insurance":
    st.header("🏠 Home Insurance – Premium Prediction")

    with st.form("home_form"):
        Claim_3_Years = "YES"
        Emp = st.selectbox("Owner Currently Employed?",["YES","NO"])
        AD1 = st.selectbox("Accidental Damage?",["YES","NO"])
        Sex = "M"
        Alarm = "NO"
        Locks = "YES"
        Bedroom = st.number_input("No of Bedrooms",min_value = 1,max_value = 6,value = 2)
        Flooding = "NO"
        Safe = "YES"
        Year_Built = st.number_input("Year Property was built",min_value=1800,max_value=2025,value=2000)

        submitted = st.form_submit_button("Predict Home Premium")

        if submitted:
            home_features = {"Claim_3_Years" : Claim_3_Years,
                "Owner_Employment_Status" : Emp,
                "Accidental_Damage" : AD1,
                "Owner_Sex" : Sex,
                "Alarm_Present" : Alarm,
                "Locks_Present" : Locks,
                "Bedrooms" : Bedroom,
                "Flooding" : Flooding,
                "Safe_Installed" : Safe,
                "YearBuilt" : Year_Built
            }

            premium = Home_Insurance.Predict("Home_Insurance/Linear_Regression.pkl","Home_Insurance/Feature_names.pkl")
            val = premium.predict_price(home_features)

            st.success(f"Estimated Home Insurance Premium: **${val:,.2f}**")

elif pills == "Health Insurance":
    st.header("🩺 Health Insurance – Premium Prediction")

    with st.form("health_form"):
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        sex = st.selectbox("Sex",["Male", "Female"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0)
        num_children = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
        smoker = st.selectbox("Smoking",["Yes","No"])
        region = st.selectbox("Region",["Northwest","Northeast","Southwest","Southeast"]) 

        submitted = st.form_submit_button("Predict Health Premium")

        if submitted:
            premium = Health_Insurance.result(age = age, sex = sex.lower(), bmi = bmi, children = num_children, smoker = smoker.lower(), region = region.lower())
            val = premium.predict() / 12
            val = float(val[0])
            st.success(f"Estimated Health Insurance Premium: **${val:,.2f}**")

