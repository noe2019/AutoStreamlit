from operator import index
import streamlit as st
import plotly.express as px
from pycaret.classification import setup as cls_setup, compare_models as cls_compare_models, pull as cls_pull, save_model, load_model
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, pull as reg_pull
from pycaret.clustering import setup as clu_setup, create_model as clu_create_model, pull as clu_pull
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 

# Load existing dataset if it exists
if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

# Sidebar for navigation and options
with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Download"])
    st.info("This application helps you explore your data and build your ML model.")

# Upload Section
if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

# Profiling Section
if choice == "Profiling" and 'df' in locals(): 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

# Modelling Section
if choice == "Modelling" and 'df' in locals(): 
    st.title("Modeling")
    st.write("Select the type of modeling task:")
    task = st.selectbox("Choose Task Type", ["Classification", "Regression", "Clustering"])

    chosen_target = None
    if task != "Clustering":
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
    
    if st.button('Run Modelling'):
        if task == "Classification":
            cls_setup(df, target=chosen_target, verbose=False)
            setup_df = cls_pull()
            st.dataframe(setup_df)
            best_model = cls_compare_models()
            compare_df = cls_pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_classification_model')
        
        elif task == "Regression":
            reg_setup(df, target=chosen_target, verbose=False)
            setup_df = reg_pull()
            st.dataframe(setup_df)
            best_model = reg_compare_models()
            compare_df = reg_pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_regression_model')
        
        elif task == "Clustering":
            clu_setup(df, session_id=123, verbose=False)
            setup_df = clu_pull()
            st.dataframe(setup_df)
            best_model = clu_create_model('kmeans')  # Example using KMeans clustering
            compare_df = clu_pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_clustering_model')

# Download Section
if choice == "Download": 
    model_file = None
    if 'best_classification_model' in os.listdir():
        model_file = 'best_classification_model.pkl'
    elif 'best_regression_model' in os.listdir():
        model_file = 'best_regression_model.pkl'
    elif 'best_clustering_model' in os.listdir():
        model_file = 'best_clustering_model.pkl'
    
    if model_file:
        with open(model_file, 'rb') as f: 
            st.download_button('Download Model', f, file_name=model_file)