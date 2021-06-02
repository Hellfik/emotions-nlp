# Sentiment Analysis 

[![Generic badge](https://img.shields.io/badge/MachineLearning-success)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/DeepLearning-success)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Python-success)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/NLP-success)](https://shields.io/)

## Sentiment analysis study

### Datasets

For this analysis, we used two differents datasets. Afterwards, we combined both to make predictions :

[Dataset link (Kaggle)](https://www.kaggle.com/ishantjuyal/emotions-in-text)

[Dataset link (Data world)](https://data.world/crowdflower/sentiment-analysis-in-text)

### Objectives

Create different classifier models and evaluate them based on criterias.
The first objective was to apply some natural language preprocessing as cleaning text, removing stopwords and using stemming or lemmatisation.
The next step was to build classifiers models and evaluate them with metrics such as the accuracy, f1-score, roc curve and the confusion matrix.

We had to display all the results on a dashboard using Streamlit (Fast building web interfaces with Python)

Classifier used :
* LogisticRegression()
* RandomForestClassifier()
* DecisionTreeClassifier()
* Neural Network ( Keras Sequential())

### Used Technologies

* Python
* Seaborn
* Pandas
* Keras
* Streamlit
* TensorFlow
* Sci-kit Learn 

### File Structure
```
├── app
│   ├── app.py
│   ├── apps
│   │   ├── best_model.h5
│   │   ├── data.py
│   │   ├── home.py
│   │   ├── model_autokeras
│   │   │   ├── assets
│   │   │   ├── saved_model.pb
│   │   │   └── variables
│   │   │       ├── variables.data-00000-of-00001
│   │   │       └── variables.index
│   │   ├── model_copy.py
│   │   ├── model.py
│   │   ├── __pycache__
│   │   │   ├── data.cpython-37.pyc
│   │   │   ├── home.cpython-37.pyc
│   │   │   ├── model_copy.cpython-37.pyc
│   │   │   └── model.cpython-37.pyc
│   │   └── text_classifier
│   │       ├── best_model
│   │       │   ├── assets
│   │       │   ├── saved_model.pb
│   │       │   └── variables
│   │       │       ├── variables.data-00000-of-00001
│   │       │       └── variables.index
│   │       ├── best_pipeline
│   │       ├── graph
│   │       ├── oracle.json
│   │       ├── trial_eb01761fad6ded6fa57c5fc8ab2d9082
│   │       │   ├── checkpoints
│   │       │   │   ├── epoch_0
│   │       │   │   │   ├── checkpoint
│   │       │   │   │   ├── checkpoint.data-00000-of-00001
│   │       │   │   │   └── checkpoint.index
│   │       │   │   └── epoch_1
│   │       │   │       ├── checkpoint
│   │       │   │       ├── checkpoint.data-00000-of-00001
│   │       │   │       └── checkpoint.index
│   │       │   ├── pipeline
│   │       │   └── trial.json
│   │       └── tuner0.json
│   ├── models
│   │   ├── dt_combined_count.sav
│   │   ├── dt_combined_tfidf.sav
│   │   ├── dt_data_world_count.sav
│   │   ├── dt_data_world_tfidf.sav
│   │   ├── dt_kaggle_count.sav
│   │   ├── dt_kaggle_tfidf.sav
│   │   ├── lr_combined_count.sav
│   │   ├── lr_combined_tfidf.sav
│   │   ├── lr_data_world_count.sav
│   │   ├── lr_data_world_tfidf.sav
│   │   ├── lr_kaggle_count.sav
│   │   ├── lr_kaggle_tfidf.sav
│   │   ├── rf_combined_count.sav
│   │   ├── rf_combined_tfidf.sav
│   │   ├── rf_data_world_count.sav
│   │   ├── rf_data_world_tfidf.sav
│   │   ├── rf_kaggle_count.sav
│   │   └── rf_kaggle_tfidf.sav
│   ├── multiapp.py
│   ├── ngrok
│   ├── __pycache__
│   │   └── multiapp.cpython-37.pyc
│   └── stream.ipynb
├── configs
├── Data
│   ├── freq_words
│   │   ├── most_freq_word_combined.csv
│   │   ├── most_freq_word_dw.csv
│   │   └── most_freq_word_k.csv
│   ├── intermediate_data
│   │   ├── clean_data2.csv
│   │   ├── clean_data_combined.csv
│   │   └── clean_data.csv
│   └── raw_data
│       ├── Emotion_final.csv
│       └── text_emotion.csv
├── __init__.py
├── notebook
│   ├── data_analysis.ipynb
│   ├── kmeans_model.ipynb
│   ├── logistic_regression_model.ipynb
│   ├── neurons_model.ipynb
│   ├── random_forest_model.ipynb
│   └── xgboost_model.ipynb
├── README.md
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-37.pyc
│   │   └── __init__.cpython-38.pyc
│   └── utils
│       ├── functions.py
│       ├── __init__.py
│       └── __pycache__
│           ├── functions.cpython-37.pyc
│           ├── functions.cpython-38.pyc
│           ├── __init__.cpython-37.pyc
│           └── __init__.cpython-38.pyc
└── test_autokeras
    ├── image_classifier
    │   ├── best_model
    │   │   ├── assets
    │   │   ├── saved_model.pb
    │   │   └── variables
    │   │       ├── variables.data-00000-of-00001
    │   │       └── variables.index
    │   ├── best_pipeline
    │   ├── graph
    │   ├── oracle.json
    │   ├── trial_b5a17c6083c8c3c8a56ee19387b69cb5
    │   │   ├── checkpoints
    │   │   │   └── epoch_0
    │   │   │       ├── checkpoint
    │   │   │       ├── checkpoint.data-00000-of-00001
    │   │   │       └── checkpoint.index
    │   │   ├── pipeline
    │   │   └── trial.json
    │   └── tuner0.json
    ├── model_autokeras
    │   ├── assets
    │   ├── saved_model.pb
    │   └── variables
    │       ├── variables.data-00000-of-00001
    │       └── variables.index
    └── test.py
```
