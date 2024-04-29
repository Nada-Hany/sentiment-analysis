import streamlit as st
import pandas as pd
import requests, joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import PIL as image 
import streamlit_lottie as st_lottie
from streamlit_option_menu import option_menu


st.set_page_config(
    page_title= 'Sentiment Analysis',
    page_icon=':gem:',
    initial_sidebar_state='collapsed'    
)

def load_lottie(url):
    r = requests.get(url)
    #200 code represents the OK status -> request was successful 
    if r.status_code != 200:
        return None
    return r.json()

model = joblib.load(open('classifier', 'rb'))

#makes a row of needed data for the model to be added to the dataset so we can predict the outcome [virtually]
#->doesn't actually get added to the dataset 
def predict(text):
    #whole row or just text ?, text/topic ? 
    #to convert it to a horizontal row 
    features = np.array([text]).reshape(1, -1)
    prediction = model.predict(features)
    return prediction


with st.sidebar:
    choose = option_menu(
        None, ['Home', 'Graph', 'Evaluation'],
        icons=['house', 'kanban', 'white_check_mark'],
        menu_icon="app-indicator", default_index=0,
        styles={
            "container":{"padding" :"5!important", "background-color":"#fafafa"},
            "icons": {"colors": "#E0E0Ef", "font-size": "25px"},
            "nav-link" : {"font-size":"16px", "text-align":"left", "margin":"0px", "--hover-color":"#eee"},
            "nav-link-selected": {"background-color":"#0e1117"},
        }
    )

if choose == 'Home':
    st.write("#Sentiment Analysis")
    st.subheader("Enter details to classify the text")
    #user input     
    

elif choose == 'Graph':
    pass

elif choose == 'Evaluation':
    pass