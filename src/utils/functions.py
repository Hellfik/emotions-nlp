from nltk.corpus import stopwords
import pandas as pd
import re
import string
import nltk
from nltk.stem import PorterStemmer
import numpy as np

# Load the Dataset

def load_dataset(file):
    '''Function to load the dataset'''
    df = pd.read_csv(file)
    return df

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

def cv(data, ngram = 1, MAX_NB_WORDS = 50000):
    count_vectorizer = CountVectorizer(ngram_range = (ngram, ngram), nax_features = MAX_NB_WORDS)
    emb = count_vectorizer.fit_transform(data).toarray()

    print("count vectorize with", str(np.array(emb).shape[1]), "features")
    return emb, count_vectorizer