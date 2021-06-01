import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.models import Model
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
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
    
    X = df['final_text'].values
    Y = df['binary_emotion'].values

    model = st.sidebar.radio("Models", ['LogisticRegression', 'Random Forest', 'Decision Tree' , 'Neural Network'])
    if model.lower() == 'logisticregression':
        vectorizer = st.sidebar.radio("Vectorizer", ['CountVectorizer', 'Tfidf', 'Word embedding'])
        if vectorizer.lower() == 'countvectorizer':
            st.header('Model performance : LogisticRegression')
            st.header('Vectorizer : CountVectorizer')
            st.sidebar.header("Hyperparametres")
            c = st.sidebar.slider('C', min_value=1, max_value=10)
            if st.button('Start Training'):
                with st.spinner(text='Training in progress'):
                    if dataset.lower() == 'data world':
                        filename = './models/lr_data_world_count.sav'
                    elif dataset.lower() == 'combined dataset':
                        filename = './models/lr_combined_count.sav'
                    else:
                        filename = './models/lr_kaggle_count.sav'
                    build_model(pickle.load(open(filename, 'rb')), X, Y, cv, True)
                    st.success('Done')
        if vectorizer.lower() == 'tfidf':
            st.header('Model performance : LogisticRegression')
            st.header('Vectorizer : Tfidf')
            st.sidebar.header("Hyperparametres")
            c = st.sidebar.slider('C', min_value=1, max_value=10)
            if st.button('Start Training'):
                with st.spinner(text='Training in progress'):
                    if dataset.lower() == 'data world':
                        filename = './models/lr_data_world_tfidf.sav'
                    elif dataset.lower() == 'combined dataset':
                        filename = './models/lr_combined_tfidf.sav'
                    else:
                        filename = './models/lr_kaggle_tfidf.sav'
                    build_model(pickle.load(open(filename, 'rb')), X, Y, tfidf_vectorizer, True)
                    st.success('Done')
    if model.lower() == 'random forest':
        vectorizer = st.sidebar.radio("Vectorizer", ['CountVectorizer', 'Tfidf', 'Word embedding'])
        if vectorizer.lower() == 'countvectorizer':
            st.header('Model performance : Random Forest Classifier')
            st.header('Vectorizer : CountVectorizer')
            st.sidebar.header("Hyperparametres")
            n_estimators = st.sidebar.slider('n_estimators', min_value=1, max_value=100)
            max_depth = st.sidebar.slider('max_depth', min_value=1, max_value=10)
            if st.button('Start Training'):
                with st.spinner(text='Training in progress'):
                    if dataset.lower() == 'data world':
                        filename = './models/rf_data_world_count.sav'
                    elif dataset.lower() == 'combined dataset':
                        filename = './models/rf_combined_count.sav'
                    else:
                        filename = './models/rf_kaggle_count.sav'
                    build_model(pickle.load(open(filename, 'rb')), X, Y, cv, False)
                    st.success('Done')
        if vectorizer.lower() == 'tfidf':
            st.header('Model performance : Random Forest classifier')
            st.header('Vectorizer : Tfidf')
            st.sidebar.header("Hyperparametres")
            n_estimators = st.sidebar.slider('n_estimators', min_value=1, max_value=100)
            max_depth = st.sidebar.slider('max_depth', min_value=1, max_value=10)
            if st.button('Start Training'):
                with st.spinner(text='Training in progress'):
                    if dataset.lower() == 'data world':
                        filename = './models/rf_data_world_tfidf.sav'
                    elif dataset.lower() == 'combined dataset':
                        filename = './models/rf_combined_tfidf.sav'
                    else:
                        filename = './models/rf_kaggle_tfidf.sav'
                    build_model(pickle.load(open(filename, 'rb')), X, Y, tfidf_vectorizer, False)
                    st.success('Done')
    if model.lower() == 'decision tree':
        vectorizer = st.sidebar.radio("Vectorizer", ['CountVectorizer', 'Tfidf', 'Word embedding'])
        if vectorizer.lower() == 'countvectorizer':
            st.header('Model performance : Decision Tree')
            st.header('Vectorizer : CountVectorizer')
            st.sidebar.header("Hyperparametres")
            c = st.sidebar.slider('C', min_value=1, max_value=10)
            if st.button('Start Training'):
                with st.spinner(text='Training in progress'):
                    if dataset.lower() == 'data world':
                        filename = './models/dt_data_world_count.sav'
                    elif dataset.lower() == 'combined dataset':
                        filename = './models/dt_combined_count.sav'
                    else:
                        filename = './models/dt_kaggle_count.sav'
                    build_model(pickle.load(open(filename, 'rb')), X, Y, cv, False)
                    st.success('Done')
        if vectorizer.lower() == 'tfidf':
            st.header('Model performance : Decision Tree')
            st.header('Vectorizer : Tfidf')
            st.sidebar.header("Hyperparametres")
            c = st.sidebar.slider('C', min_value=1, max_value=10)
            if st.button('Start Training'):
                with st.spinner(text='Training in progress'):
                    if dataset.lower() == 'data world':
                        filename = './models/dt_data_world_tfidf.sav'
                    elif dataset.lower() == 'combined dataset':
                        filename = './models/dt_combined_tfidf.sav'
                    else:
                        filename = './models/dt_kaggle_tfidf.sav'
                    build_model(pickle.load(open(filename, 'rb')), X, Y, tfidf_vectorizer, False)
                    st.success('Done')
    if model.lower() == 'neural network':
        vectorizer = st.sidebar.radio("Vectorizer", ['CountVectorizer', 'Tfidf', 'Word embedding'])
        if vectorizer.lower() == 'countvectorizer':
            st.header('Model performance : Neural network classifier')
            st.header('Vectorizer : CountVectorizer')
            st.sidebar.header("Hyperparametres")
            if st.button('Start Training'):
                with st.spinner(text='Training in progress'):
                    create_nn_model(X,Y,cv)
                    #compilation_model(build_neural_network(X, Y), X, Y)
                    #auto_keras(df['final_text'].values, df['binary_emotion'])
                    st.success('Done')
        if vectorizer.lower() == 'tfidf':
            st.header('Model performance : Neural network classifier')
            st.header('Vectorizer : Tfidf')
            st.sidebar.header("Hyperparametres")
            if st.button('Start Training'):
                with st.spinner(text='Training in progress'):
                    create_nn_model(X,Y,tfidf_vectorizer)
                    #compilation_model(build_neural_network(X, Y), X, Y)
                    #auto_keras(df['final_text'].values, df['binary_emotion'])
                    st.success('Done')
        if vectorizer.lower() == 'word embedding':
            st.header('Model performance : Neural network classifier')
            st.header('Vectorizer : Word Embedding')
            st.sidebar.header("Hyperparametres")
            if st.button('Start Training'):
                with st.spinner(text='Training in progress'):
                    compilation_model(build_neural_network(X, Y), X, Y)
                    #auto_keras(X, Y)
                    #create_neural_network_emb(X,Y)
                    st.success('Done')
def build_model(model, X, Y, vectorizer, decision):
    labels = [0,1]
    X, countvectorizer = vectorizer(X)

    # Model building
 
    # Splitting the data into training set and testing set
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)
    # Model
    clf = model
    # Fitting the model
    #clf.fit(X_train, Y_train)
    # Saving the model
    #file_name = filename
    #pickle.dump(clf, open(file_name, 'wb'))
    # Accuracy score
    score = clf.score(X_test, Y_test)
    st.write('Accuracy:', score)
    # Making predictions
    ypred = clf.predict(X_test)
    
    sep()
    # Showing classification report
    st.header('Classification Report')
    sep()

    report_dict = classification_report(Y_test, ypred, output_dict=True)
    st.dataframe(report_dict).T

    sep()
    # Showing confusion matrix
    st.header('Confusion Matrix')
    sep()
    cm = confusion_matrix(Y_test, ypred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    st.pyplot()
    if decision:
        y_score = model.decision_function(X_test)
        average_precision = average_precision_score(Y_test, y_score)
        disp = plot_precision_recall_curve(model, X_test, Y_test)
        disp.ax_.set_title('2-class Precision-Recall curve: '
                        'AP={0:0.2f}'.format(average_precision))
        st.pyplot()

def sep():
    st.markdown('''--------------------------''')


#-------------------------------#
#   Neural network functions    #
#-------------------------------#

# Tokenizing data for the neural network inputs
def preparing_input_features(X):
    max_len = 1000
    tok = Tokenizer(num_words=2000)
    tok.fit_on_texts(X)
    sequences = tok.texts_to_sequences(X)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

    return sequences_matrix, max_len

# Neural network model
def build_neural_network(X, Y):
    sequences_matrix, max_len = preparing_input_features(X)

    X_train, X_test, Y_train, Y_test = train_test_split(sequences_matrix, Y, test_size=0.3, random_state=2)

    inputs = Input(name='inputs',shape=[max_len])#step1
    layer = Embedding(2000,50,input_length=max_len)(inputs) #step2
    layer = Activation('relu')(layer) # step3
    layer = Dropout(0.2)(layer) # step4
    layer = Dense(1,name='out_layer')(layer) #step5 again but this time its giving only one output as because we need to classify the tweet as positive or negative
    layer = Activation('sigmoid')(layer) #step6 but this time activation function is sigmoid for only one output.
    model = Model(inputs=inputs,outputs=layer) #here we are getting the final output value in the model for classification
    

    return model, X_test, Y_test, X_train, Y_train #function returning the value when we call it

def compilation_model(model, X, Y):
    # Here we are calling the function of created model
    model, X_test, Y_test, X_train, Y_train = build_neural_network(X, Y)
    model.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])  

    history = model.fit(X_train,Y_train,batch_size=100,epochs=2, validation_split=0.1)
    score = model.evaluate(X_test, Y_test)
    st.write('Accuracy:',score[1])
    # Show the learning curves
    history_df = pd.DataFrame(history.history)
    st.write(history_df)
    history_df.loc[:, ['loss', 'val_loss']].plot()
    st.pyplot()

# Autokeras model to find the best neural network structure(neurons and layers)
def auto_keras(variables, target):
    X = variables
    y = target

    x_train ,x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = tf.keras.models.load_model('/work/emotions-nlp/app/apps/model_autokeras', custom_objects=ak.CUSTOM_OBJECTS)
    #clf.fit(x_train, y_train)
    # Predict with the best model.
    predicted_y = clf.predict(tf.expand_dims(x_test, -1))
    # Evaluate the best model with testing data.
    #st.write('Accuracy:',(clf.evaluate(tf.expand_dims(x_test, -1), tf.expand_dims(y_test, -1))[1]))
    # Show the learning curves
    st.write(predicted_y)

def create_neural_network(X_train, y_train, X_test, y_test):
    input_dim = X_train.shape[1]
    model = keras.Sequential([
    layers.Dense(512, input_shape=(input_dim,), activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(2, activation='sigmoid')
    ])
    model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics = 'accuracy',
    )
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

    st.write('Accuracy:',score[1])
    history_df = pd.DataFrame(history.history)

    return history_df


def display_nn(history_df):
    # Show the learning curves
    history_df.loc[:, ['loss', 'val_loss']].plot()
    st.pyplot()

def create_nn_model(X,y,vectorizer):
    X, countvectorizer = vectorizer(X)
    x_train ,x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    history_df = create_neural_network(x_train, y_train, x_test, y_test)
    display_nn(history_df)

def create_neural_network_emb(X,y):

    sequence_matrix, max_length = preparing_input_features(X)

    X_train ,X_test, y_train, y_test = train_test_split(sequence_matrix, y, test_size=0.2, random_state=42)

    model = keras.Sequential([
    layers.Embedding(2000,50, input_length=max_length),
    layers.Dense(512, input_shape=[max_length], activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(2, activation='sigmoid')
    ])
    model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics = 'accuracy',
    )
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

    st.write('Accuracy:',score[1])
    history_df = pd.DataFrame(history.history)

    display_nn(history_df)
    
    """pred = pd.DataFrame(model.predict(X_test))
    pred.rename(columns={0: "Negatif", 1: "Positif"}, errors="raise", inplace=True)
    sentences_df = pd.DataFrame(X_test)
    sentences_pred = pd.concat([sentences_df, pred], axis=1)"""