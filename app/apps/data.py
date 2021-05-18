import streamlit as st
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, "/work/emotions-nlp")

from src.utils.functions import load_dataset


def sidebar():
    st.sidebar.markdown("<h1 style='text-align: center; color: #DF362D; font-size:30px !importnt'>Sentiment Analysis</h1>", unsafe_allow_html=True)
    st.sidebar.info('😀 Welcome to our sentiment Analysis 😀')
    number_display = st.sidebar.slider('Number of results', min_value=5, max_value=500)
    choice = st.sidebar.radio("Dataframe", ['Original', 'Formated', 'Tokenized', 'Stopwords', 'Stemming'])
    df = show_clean_data('/work/emotions-nlp/Data/intermediate_data/clean_data.csv', choice)
    st.dataframe(df.head(number_display))


def app():
    st.title('Data Page')

    st.write("This is the `Data` page of the Sentiment Analysis.")
    st.markdown('''
        For this analysis, we will be using a dataset from [Kaggle](https://www.kaggle.com/ishantjuyal/emotions-in-text)
    ''')

    sidebar()

def show_clean_data(path, option = "original"):
    df = load_dataset(path)
    st.markdown('''-----------------------------------------------''')
    if option.lower() == "formated":
        st.markdown('''<h2 style="font-style:italic">Formated Data (text to lower case, removed special characters, removed links, ponctuation and contractions handling)</h2>''', unsafe_allow_html=True)
        df = df[['Text', 'Emotion', 'clean_text']]
    elif option.lower() == "tokenized":
        st.markdown('''<h2 style="font-style:italic">Tokenized Data</h2>''', unsafe_allow_html=True)
        df = df = df[['Text', 'Emotion', 'clean_text', 'tokenized']]
    elif option.lower() == "stopwords":
        st.markdown('''<h2 style="font-style:italic">Stopwrods removing</h2>''', unsafe_allow_html=True)
        df = df = df[['Text', 'Emotion', 'clean_text', 'tokenized', 'stopwords_removed']]
    elif option.lower() == "stemming":
        st.markdown('''<h2 style="font-style:italic">Stemming</h2>''', unsafe_allow_html=True)
        df = df = df[['Text', 'Emotion', 'clean_text', 'tokenized', 'stopwords_removed', 'porter_stemmer']]
    else:
        st.markdown('''<h2 style="font-style:italic">Original Data</h2>''', unsafe_allow_html=True)
        df = df[['Text', 'Emotion']]
    return df


