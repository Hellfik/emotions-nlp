
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#from notebook.logistic_regression_model import *
import sys
sys.path.insert(0, "/work/emotions-nlp")

from src.utils.functions import load_dataset, cv, tfidf_vectorizer

def app():
    st.title('Model')

    st.write('This is the `Model` page of the multi-page app.')

    st.write('The model performance of the Kaggle Dataset is presented below.')

    sidebar()


def sidebar():
    # Load Dataset
    df = load_dataset('/work/emotions-nlp/Data/intermediate_data/clean_data.csv')
    X = df['porter_stemmer']
    Y = df['Emotion']

    st.sidebar.markdown("<h1 style='text-align: center; color: #DF362D; font-size:30px !importnt'>Sentiment Analysis</h1>", unsafe_allow_html=True)
    st.sidebar.info('😀 Welcome to our sentiment Analysis 😀')
    model = st.sidebar.radio("Models", ['LogisticRegression'])
    vectorizer = st.sidebar.radio("Vectorizer", ['CountVectorizer', 'Tfidf', 'Word embedding'])
    if model.lower() == 'logisticregression':
        if vectorizer.lower() == 'countvectorizer':
            st.write('cv')
            st.header('Model performance : LogisticRegression')
            st.header('Vectorizer : CountVectorizer')
            st.sidebar.header("Hyperparametres")
            c = st.sidebar.slider('C', min_value=1, max_value=10)
            if st.button('Start Training'):
                with st.spinner(text='Training in progress'):
                    build_model(LogisticRegression(C=c), X, Y, cv)
                    st.success('Done')
        if vectorizer.lower() == 'tfidf':
            st.write('tfidf')
            st.header('Model performance : LogisticRegression')
            st.header('Vectorizer : Tfidf')
            st.sidebar.header("Hyperparametres")
            c = st.sidebar.slider('C', min_value=1, max_value=10)
            if st.button('Start Training'):
                with st.spinner(text='Training in progress'):
                    build_model(LogisticRegression(C=c), X, Y, tfidf_vectorizer)
                    st.success('Done')
    if model.lower() == 'neural network':
        st.header('Model performance : Neural Network')
        build_model(LogisticRegression(), X, Y)

def build_model(model, X, Y, vectorizer):
    # Encoding target variable
    le = preprocessing.LabelEncoder()
    Y = le.fit_transform(Y.values)
    
    X, countvectorizer = vectorizer(X)

    # Model building
 
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)
    clf = model
    clf.fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)
    st.write('Accuracy:')
    st.write(score)