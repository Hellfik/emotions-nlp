#-------------------------------#
#         Module import         #
#-------------------------------#
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
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
#import autokeras as ak

import sys
sys.path.insert(0, "/work/emotions-nlp")

from src.utils.functions import load_dataset, cv, tfidf_vectorizer


#----------------------------------------------#
#   Main function that calls our components    #
#----------------------------------------------#

def app():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Model')

    st.write('This is the `Model` page of the multi-page app.')

    st.write('The model performance of the Kaggle Dataset is presented below.')

    sidebar()

#-------------------------------#
#           Sidebar             #
#-------------------------------#

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

    model = st.sidebar.radio("Models", ['LogisticRegression', 'Random Forest', "XGBoost" , 'Neural Network'])
    if model.lower() == 'logisticregression':
        vectorizer = st.sidebar.radio("Vectorizer", ['CountVectorizer', 'Tfidf', 'Word embedding'])
        if vectorizer.lower() == 'countvectorizer':
            st.header('Model performance : LogisticRegression')
            st.header('Vectorizer : CountVectorizer')
            st.sidebar.header("Hyperparametres")
            c = st.sidebar.slider('C', min_value=1, max_value=10)
            if st.button('Start Training'):
                with st.spinner(text='Training in progress'):
                    build_model(LogisticRegression(C=c), X, Y, cv, True)
                    st.success('Done')
        if vectorizer.lower() == 'tfidf':
            st.header('Model performance : LogisticRegression')
            st.header('Vectorizer : Tfidf')
            st.sidebar.header("Hyperparametres")
            c = st.sidebar.slider('C', min_value=1, max_value=10)
            if st.button('Start Training'):
                with st.spinner(text='Training in progress'):
                    build_model(LogisticRegression(C=c), X, Y, tfidf_vectorizer, True)
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
                    build_model(RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth), X, Y, cv, False)
                    st.success('Done')
        if vectorizer.lower() == 'tfidf':
            st.header('Model performance : Random Forest classifier')
            st.header('Vectorizer : Tfidf')
            st.sidebar.header("Hyperparametres")
            n_estimators = st.sidebar.slider('n_estimators', min_value=1, max_value=100)
            max_depth = st.sidebar.slider('max_depth', min_value=1, max_value=10)
            if st.button('Start Training'):
                with st.spinner(text='Training in progress'):
                    build_model(RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth), X, Y, tfidf_vectorizer, False)
                    st.success('Done')
    if model.lower() == 'xgboost':
        vectorizer = st.sidebar.radio("Vectorizer", ['CountVectorizer', 'Tfidf', 'Word embedding'])
        if vectorizer.lower() == 'countvectorizer':
            st.header('Model performance : XGBoost')
            st.header('Vectorizer : CountVectorizer')
            st.sidebar.header("Hyperparametres")
            c = st.sidebar.slider('C', min_value=1, max_value=10)
            if st.button('Start Training'):
                with st.spinner(text='Training in progress'):
                    build_model(XGBClassifier(random_state=0,booster="dart",learning_rate=0.02,n_estimators=200,n_jobs=-1, objective='binary:logistic',use_label_encoder=False,eval_metric="auc"), X, Y, cv, False)
                    st.success('Done')
        if vectorizer.lower() == 'tfidf':
            st.header('Model performance : XGBoost')
            st.header('Vectorizer : Tfidf')
            st.sidebar.header("Hyperparametres")
            c = st.sidebar.slider('C', min_value=1, max_value=10)
            if st.button('Start Training'):
                with st.spinner(text='Training in progress'):
                    build_model(XGBClassifier(random_state=0,booster="dart",learning_rate=0.02,n_estimators=200,n_jobs=-1, objective='binary:logistic',use_label_encoder=False,eval_metric="auc"), X, Y, tfidf_vectorizer, False)
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

#----------------------------------------#
#   Machine learning models functions    #
#----------------------------------------#

def build_model(model, X, Y, vectorizer, decision):
    """ 
    Function to build a model based on the user choice: it displays the accuracy, the classification report and the confusion matrix
    {model} => Define the machine learning model to use for the training
    {X} => Text corpus variable
    {Y} => Target variable (class to predict)
    {vectorizer} => Technique that will be used to transform the text corpus into vectors (countvectorizer, tfdif)
    {decision} => Boolean on which model we can use the decision_function()
    """
    labels = [0,1]
    X, countvectorizer = vectorizer(X)

    # Model building
 
    # Splitting the data into training set and testing set
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)
    # Model
    clf = model
    # Fitting the model
    clf.fit(X_train, Y_train)
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
    """
    Simple function to creates separators in the Streamlit Layout
    """
    st.markdown('''--------------------------''')


#-------------------------------#
#   Neural network functions    #
#-------------------------------#

def preparing_input_features(X):
    """
    Tokenizing data for the neural network inputs, this opertation is needed before we
    can train our neural network model
    It takes as parameter the text variable we are trying to classify as positive or negative
    {X} => text corpus paramater
    Returns the generated matrix and the features length
    """
    max_len = 1000
    tok = Tokenizer(num_words=2000)
    tok.fit_on_texts(X)
    sequences = tok.texts_to_sequences(X)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

    return sequences_matrix, max_len


# Autokeras model to find the best neural network structure(neurons and layers)
def auto_keras(variables, target):
    """
    Function that finds the best structure for our neural network model
    {variable} => Data to train on
    {target} => The target variable (What we are trying to predict)
    """
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
    """
    Function that creates a neural network using keras Sequential object
    Activation function => Rectified Linear Unit (Return 0 if a negative value is passed and the value back if its > 0)
    Last layer activation function => Sigmoid as we are trying to predict two classes (This will give us the probability of getting 1)
    The metric that we will be evaluting our model on is the accuracy
    Returns a dataframe that will be used to plot the learning curve
    """
    input_dim = X_train.shape[1]
    model = keras.Sequential([
    layers.Dense(516, input_shape=(input_dim,), activation='relu'),
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
    """
    Function to display the learning curve of the neural network
    {history_df} => Dataframe parameter
    """
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