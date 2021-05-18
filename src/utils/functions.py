from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Load the Dataset

def load_dataset(file):
    '''Function to load the dataset'''
    df = pd.read_csv(file)
    return df

#-------------------------------#
#       Data preprocessing
#-------------------------------#

# Applying a first round of text cleaning techniques
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = text.replace(r"(can't|cannot)", 'can not')
    text = text.replace(r"(didn't|didnt)", 'did not')
    text = text.replace(r"n't", ' not')
    return text

# Remove English StopWords
def remove_stopwords(text):
    """
    Removing stopwords belonging to english language
    
    """
    stopword = stopwords.words('english')
    stopword.remove('not')
    stopword.remove('nor')
    stopword.remove('no')
    
    words = [word for word in text if word not in stopword]
    return words

# Stemming Operation
def porter_stemmer(text):
    """
        Stem words in list of tokenized words with PorterStemmer
    """
    stemmer = nltk.PorterStemmer()
    stems = [stemmer.stem(i) for i in text]
    return stems

#-------------------------------#
#          Vectorizer
#-------------------------------#

def cv(data, ngram = 1, MAX_NB_WORDS = 1000):
    count_vectorizer = CountVectorizer(ngram_range = (ngram, ngram), max_features = MAX_NB_WORDS)
    emb = count_vectorizer.fit_transform(data).toarray()

    print("count vectorize with", str(np.array(emb).shape[1]), "features")
    return emb, count_vectorizer

def tfidf_vectorizer(data, MAX_NB_WORDS = 1000):
    tfv = TfidfVectorizer(max_features=MAX_NB_WORDS, 
            strip_accents='unicode', analyzer='word',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1)
    emb = tfv.fit_transform(data).toarray()

    print("Tfidf vectorize with", str(np.array(emb).shape[1]), "features")
    return emb, tfv

#-------------------------------#
# Logistic Regression Evaluation
#-------------------------------#

# Return the matrix of confusion for the choosen model
def get_confusion_matrix(X_test, y_test, model):
    matrix_df = pd.DataFrame(
        data = confusion_matrix(y_true = y_test, y_pred = model.predict(X_test)),
        index = model.classes_ + " actual",
        columns = model.classes_ + " predicted").T
    return matrix_df

# Heatmap of the matrix of confusion
def show_matrix_heatmap(matrix):
    for i in matrix:
        plt.title("Heatmap for the confusion matrix")
        sns.heatmap(i, annot=True, fmt='d', cmap='YlGnBu')
        plt.show()

# Function to plot the roc curve
def show_roc_curve(X_test, y_test, model):
    #define metrics
    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba, pos_label='Yes')
    idx = np.min(np.where(tpr > 0.95)) 
    
    #create ROC curve
    plt.plot(fpr,tpr,label="AUC="+str(round((roc),3)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='red')
    plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='red')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)

    plt.show()