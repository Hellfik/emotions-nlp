import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
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

    st.sidebar.markdown("<h1 style='text-align: center; color: #DF362D; font-size:30px !importnt'>Sentiment Analysis</h1>", unsafe_allow_html=True)
    st.sidebar.info('😀 Welcome to our sentiment Analysis 😀')
    dataset = st.sidebar.selectbox('Which Dataset do you want to use?',('Kaggle', 'Data world', 'Combined Dataset'))
    
    if dataset.lower() == 'data world':
        df = load_dataset('/work/emotions-nlp/Data/intermediate_data/clean_data2.csv')
    elif dataset.lower() == 'combined dataset':
        df = load_dataset('/work/emotions-nlp/Data/intermediate_data/clean_data_combined.csv')
    else:
        df = load_dataset('/work/emotions-nlp/Data/intermediate_data/clean_data.csv')
    
    X = df['porter_stemmer']
    Y = df['binary_emotion']

    model = st.sidebar.radio("Models", ['LogisticRegression', 'Random Forest', "Neural Network"])
    vectorizer = st.sidebar.radio("Vectorizer", ['CountVectorizer', 'Tfidf', 'Word embedding'])
    if model.lower() == 'logisticregression':
        if vectorizer.lower() == 'countvectorizer':
            st.header('Model performance : LogisticRegression')
            st.header('Vectorizer : CountVectorizer')
            st.sidebar.header("Hyperparametres")
            c = st.sidebar.slider('C', min_value=1, max_value=10)
            if st.button('Start Training'):
                with st.spinner(text='Training in progress'):
                    build_model(LogisticRegression(C=c), X, Y, cv)
                    st.success('Done')
        if vectorizer.lower() == 'tfidf':
            st.header('Model performance : LogisticRegression')
            st.header('Vectorizer : Tfidf')
            st.sidebar.header("Hyperparametres")
            c = st.sidebar.slider('C', min_value=1, max_value=10)
            if st.button('Start Training'):
                with st.spinner(text='Training in progress'):
                    build_model(LogisticRegression(C=c), X, Y, tfidf_vectorizer)
                    st.success('Done')
    if model.lower() == 'random forest':
        if vectorizer.lower() == 'countvectorizer':
            st.header('Model performance : Random Forest Classifier')
            st.header('Vectorizer : CountVectorizer')
            st.sidebar.header("Hyperparametres")
            n_estimators = st.sidebar.slider('n_estimators', min_value=1, max_value=100)
            max_depth = st.sidebar.slider('max_depth', min_value=1, max_value=10)
            if st.button('Start Training'):
                with st.spinner(text='Training in progress'):
                    build_model(RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth), X, Y, cv)
                    st.success('Done')
        if vectorizer.lower() == 'tfidf':
            st.header('Model performance : Random Forest classifier')
            st.header('Vectorizer : Tfidf')
            st.sidebar.header("Hyperparametres")
            n_estimators = st.sidebar.slider('n_estimators', min_value=1, max_value=100)
            max_depth = st.sidebar.slider('max_depth', min_value=1, max_value=10)
            if st.button('Start Training'):
                with st.spinner(text='Training in progress'):
                    build_model(RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth), X, Y, tfidf_vectorizer)
                    st.success('Done')
    if model.lower() == 'neural network':
        if vectorizer.lower() == 'countvectorizer':
            st.header('Model performance : Neural network lassifier')
            st.header('Vectorizer : CountVectorizer')
            st.sidebar.header("Hyperparametres")
            c = st.sidebar.slider('C', min_value=1, max_value=10)
            if st.button('Start Training'):
                with st.spinner(text='Training in progress'):
                    #build_model(RandomForestClassifier(n_estimators=10, max_depth=3), X, Y, cv)
                    st.success('Done')

def build_model(model, X, Y, vectorizer):
    # Encoding target variable
    #le = preprocessing.LabelEncoder()
    #Y = le.fit_transform(Y.values)
    labels = [0,1]
    X, countvectorizer = vectorizer(X)

    # Model building
 
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)
    clf = model
    clf.fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)
    st.write('Accuracy:', score)

    ypred = clf.predict(X_test)
    
    sep()
    st.header('Classification Report')
    sep()

    report_dict = classification_report(Y_test, ypred, output_dict=True)
    st.dataframe(report_dict).T

    sep()
    st.header('Confusion Matrix')
    sep()

    st.set_option('deprecation.showPyplotGlobalUse', False)
    cm = confusion_matrix(Y_test, ypred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    st.pyplot()


def sep():
    st.markdown('''--------------------------''')

