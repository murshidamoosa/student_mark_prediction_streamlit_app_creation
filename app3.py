import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from sklearn.linear_model import LinearRegression



st.title("Student Mark Prediction")

duration_of_study= st.number_input("enter hours of study", min_value=0, max_value=24, value=0)
class_attendence=st.number_input("enter attendence",min_value=60,max_value=100)
Access_to_Resources=st.selectbox("resources",["high","medium","low"])
motivation_level=st.selectbox("motivation",["high","medium","low"])

resource_dict = {"high":1, "medium":2, "low":3}
motivation_dict = {"high":1, "medium":2, "low":3}

resource_mapping = resource_dict[Access_to_Resources]
motivational_mapping=motivation_dict[motivation_level]

input_data={
    "Hours_Studied":duration_of_study,
    "Attendance":class_attendence,
    "Access_to_Resources_m":resource_mapping,
    "Motivation_Level_m":motivational_mapping
}

new_data=pd.DataFrame([input_data])

df= pd.read_csv("students.csv")
columns_list = [col for col in df.columns if col != 'Unnamed: 0']



new_data = new_data.reindex(columns=columns_list, fill_value=0)

with open("Linear_regression_model.pkl","rb") as regression_file:
    loaded_model=pickle.load(regression_file)
    
prediction=loaded_model.predict(new_data)


if prediction[0]<=60:
    st.error("prediction: you are failure")
else:
    st.success("prediction: you are success")