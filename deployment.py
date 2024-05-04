import streamlit as st
import pandas as pd
import requests, joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import PIL as image 
import streamlit_lottie as st_lottie
from streamlit_option_menu import option_menu
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import stopwords
import pickle

loaded_model = pickle.load(open('trained_model.sav', 'rb'))
vectroized_model = pickle.load(open('vectorized_model.sav', 'rb'))
label_encoders = pickle.load(open('label_encoder.sav', 'rb'))


def predict(text):
    # preprocess
    stop_words = set(stopwords.words('english')) 
    lemmatizer = WordNetLemmatizer() 
    translator = str.maketrans('', '', string.punctuation)  
    text = ' '.join([lemmatizer.lemmatize(word.translate(translator).lower()) for word in text.split() if word.lower() not in stop_words])
    unique_words = set(text.split())
    text = ' '.join(unique_words)

    # Make predictions using the loaded model
    features = vectroized_model.transform([text])
    prediction = loaded_model.predict(features)
    decoded_prediction = label_encoders["Sentiment (Label)"].inverse_transform(prediction)


    return decoded_prediction[0]


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


with st.sidebar.expander("Menu", expanded=True):
    choose = option_menu(None, ["Home", "Graphs"],
                         icons=['house', 'kanban', 'book',],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#0E1117"},
        "icon": {"color": '#E0E0EF', "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#0E1117",
                      "color": "#ffffff"},
        "nav-link-selected": {"background-color": "#ffffff", "color": "#0E1117"},
    }
    )

if choose=='Home':
    
    st.title("Welcome to Sentiment Analysis App")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
        # Text area for user input
    user_input = st.text_area("Enter some text below to analyze its sentiment.")

        # Button to trigger sentiment analysis
    if st.button("Analyze Sentiment"):
            if user_input:
                sentiment = predict(user_input)
                st.write("Sentiment:", sentiment)
              
            else:
                st.write("Please enter some text.")
    
elif choose == 'Graphs':
        pass