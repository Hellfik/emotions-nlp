import streamlit as st
import pandas as pd
import numpy as np
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
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import kerastuner as kt
import autokeras as ak

import sys
sys.path.insert(0, "/work/emotions-nlp")

from src.utils.functions import load_dataset, cv, tfidf_vectorizer

def app():
    st.set_option('deprecation.showPyplotGlobalUse', False)
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
    Y = df['Emotion']

    model = st.sidebar.radio("Models", ["Neural Network"])


def sep():
    st.markdown('''--------------------------''')


#-------------------------------#
#   Neural network functions    #
#-------------------------------#

def display_nn():
    st.write(sentences_pred)
    # Show the learning curves
    history_df.loc[:, ['loss', 'val_loss']].plot()
    st.pyplot()

def create_neural_network(X_train, y_train, X_test, y_test):

    input_dim = X_train.shape[1]
    model = keras.Sequential([
    layers.Dense(10, input_shape=(input_dim,), activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(10, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(10, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(6, activation='sigmoid')
    ])
    model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics = 'accuracy',
    )
    #st.write('', model.summary())
    #st.write('OK')
    history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=100,
    epochs=10,
    verbose=1
    )

    score = model.evaluate(X_test, y_test)
    early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
    )

    print('Accuracy:',score[1])
    history_df = pd.DataFrame(history.history)
    #st.write(history_df)
    
    
    pred = pd.DataFrame(model.predict(X_test))
    pred.rename(columns={0: "Negatif", 1: "Positif"}, errors="raise", inplace=True)
    sentences_df = pd.DataFrame(x_test)
    sentences_pred = pd.concat([sentences_df, pred], axis=1)
    #st.write(sentences_pred)
    return sentences_pred, history_df



def create_model(data, model):
    '''
    Create a model that takes the data and returns vectors
    '''
    if model == 'TfidfVectorizer':
        tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2))
    elif model == 'CountVectorizer':
        tf = CountVectorizer(analyzer = 'word', ngram_range = (1, 2))   

    return tf


df = load_dataset('/work/emotions-nlp/Data/intermediate_data/clean_data.csv')




X = df['final_text'].values
le = preprocessing.LabelEncoder()
y = df['binary_emotion'].values



x_train ,x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tf = create_model(df, 'CountVectorizer')

tf.fit(x_train)

x_train = tf.transform(x_train)
x_test = tf.transform(x_test)

sentences_pred, history_df = create_neural_network(x_train, y_train, x_test, y_test)



# Initialize the text classifier.
#clf = ak.TextClassifier(
#    overwrite=True, max_trials=1
#)  # It only tries 1 model as a quick demo.
# Feed the text classifier with training data.
#clf.fit(x_train, y_train, epochs=2)
# Predict with the best model.
#predicted_y = clf.predict(x_test)
# Evaluate the best model with testing data.
#print(clf.evaluate(x_test, y_test))
#model = clf.export_model()
#try:
#    model.save("model_autokeras", save_format="tf")
#except Exception:
#    model.save("model_autokeras.h5")

