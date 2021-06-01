import pandas as pd
import streamlit as st
from multiapp import MultiApp
from apps import home, data, model, model_copy

import sys
sys.path.insert(0, "/work/emotions-nlp")


app = MultiApp()

st.markdown("""
# Sentiment Analysis
This multi-page app is using the [streamlit-multiapps](https://github.com/upraneelnihar/streamlit-multiapps) framework developed by [Praneel Nihar](https://medium.com/@u.praneel.nihar). Also check out his [Medium article](https://medium.com/@u.praneel.nihar/building-multi-page-web-app-using-streamlit-7a40d55fa5b4).
Made By **Farid** et **Mickael**
""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Data", data.app)
app.add_app("Model", model_copy.app)
#app.add_app("Model2", model.app)
# The main app
app.run()