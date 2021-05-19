import streamlit as st
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown as md
sys.path.insert(0, "/work/emotions-nlp")

from src.utils.functions import load_dataset, find_most_common_word


def sidebar():
    st.sidebar.markdown("<h1 style='text-align: center; color: #DF362D; font-size:30px !importnt'>Sentiment Analysis</h1>", unsafe_allow_html=True)
    st.sidebar.info('😀 Welcome to our sentiment Analysis 😀')
    dataset = st.sidebar.selectbox('Which Dataset do you want to use?',('Kaggle', 'Data world', 'Combined Dataset'))
    number_display = st.sidebar.slider('Number of results', min_value=5, max_value=500)
    choice = st.sidebar.radio("Dataframe", ['Original', 'Formated', 'Tokenized', 'Stopwords', 'Stemming'])
    if dataset.lower() == 'data world':
        df = show_clean_data('/work/emotions-nlp/Data/intermediate_data/clean_data2.csv', choice)
        most_freq_word = load_dataset('/work/emotions-nlp/Data/freq_words/most_freq_word_dw.csv')
    elif dataset.lower() == 'combined dataset':
        df = show_clean_data('/work/emotions-nlp/Data/intermediate_data/clean_data_combined.csv', choice)
        most_freq_word = load_dataset('/work/emotions-nlp/Data/freq_words/most_freq_word_combined.csv')
    else:
        df = show_clean_data('/work/emotions-nlp/Data/intermediate_data/clean_data.csv', choice)
        most_freq_word = load_dataset('/work/emotions-nlp/Data/freq_words/most_freq_word_k.csv')
    st.dataframe(df.head(number_display))
    

    sep()
    st.header('Dataframe Infos')
    st.markdown('''
        |Number of rows |Number of columns  | Missing values|
            :---: | :---: | :---:
            |{0}|{1}|{2}|
    '''.format(df.shape[0], df.shape[1], df.isnull().sum()[1]))
    sep()

    sns.histplot(data=df, y='Emotion', hue='Emotion')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.title('Emotion Histogramme')
    st.pyplot()

    sep()
    
    sns.barplot(data=most_freq_word, x='frequence', y='word', hue='word')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.title('Word frequency Histogramme')
    st.pyplot()


def app():
    st.title('Data Page')

    st.write("This is the `Data` page of the Sentiment Analysis.")
    st.markdown('''
        For this analysis, we will be using a dataset from [Kaggle](https://www.kaggle.com/ishantjuyal/emotions-in-text)
    ''')
    st.text("Chose a Dataset and a Classifier in the sidebar. Input your values and get a prediction")
    sidebar()



def show_clean_data(path, option = "original"):
    df = load_dataset(path)
    sep()
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


def sep():
    st.markdown('''---------------------------------------------''')