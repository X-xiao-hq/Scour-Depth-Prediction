# Core Pkgs
import streamlit as st 

# EDA Pkgs
import pandas as pd 
import numpy as np 

# Utils
import os
import joblib 
import hashlib
# passlib,bcrypt

# Data Viz Pkgs
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')

# Load ML Models
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model

def main():
    """Scour Depth Prediction App"""
    st.title("Scour Depth Prediction App")
    st.subheader("Input Parameters")
    #X = df[['Rep', 'Red', KC', 'shield', ]].values
    Rep = st.number_input("Rep (Particle Reynolds Number)", min_value=0.0, value=1000.0)
    Red = st.number_input("Red (Reynolds Number)", min_value=0.0, value=10000.0)
    KC = st.number_input("KC (Keulegan-Carpenter Number)", min_value=0.0, value=10.0)
    shield = st.number_input("Shield Parameter", min_value=0.0, value=0.05)

    feature_list = [Rep, Red, KC, shield]
    single_sample = np.array(feature_list).reshape(1,-1)

    # 修正：下拉菜单选项应该与实际的模型文件对应
    model_choice = st.selectbox("Select Model",["xgb model","XGB","shap values"])
    
    if st.button("Predict"):
        st.write("Predicted scour depth wrt the diamater is : ")
        if model_choice == "xgb model":
            loaded_model = load_model("xgb_model.pkl")
            prediction = loaded_model.predict(single_sample)
            st.write(prediction[0])
        elif model_choice == "XGB":
            loaded_model = load_model("XGB.kpl")
            prediction = loaded_model.predict(single_sample)
            st.write(prediction[0])
        elif model_choice == "shap values":
            loaded_model = load_model("shap_values.pkl")
            prediction = loaded_model.predict(single_sample)
            st.write(prediction[0])

if __name__ == '__main__':
    main()
