import streamlit as st
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
from models import count_victorizer, transformer, model_lr


stemmer=PorterStemmer()
def clean(text):
    # text="".join([char for char in text if char not in string.punctuation])
    text="".join([re.sub('[^a-zA-Z]',' ',char) for char in text ])
    text=text.lower()
    text=text.split()
    text=[stemmer.stem(word) for word in text if word not in set(stopwords.words("english"))]
    text=" ".join(text)
    return text


st.title('Fake News Detector')

choice = st.selectbox("Select A Classifier", ["Logistic Regression", "SVM", "Decision Tree", "Random Forest"])

text = st.text_input("Enter News (Author And Artical Title)")

clicked = st.button("Classify")

md = "<div style='text-align: center;'>Just Give It A Try</div>"
if clicked:
    text = clean(text)
    c = count_victorizer.transform(np.array([text]))
    features = transformer.transform(c)

    prediction = model_lr.predict(features)[0]
    
    md = "<h5 style='text-align: center;color: #F00;'>" + ("Fake News" if prediction == 1 else "Real News") + "</h5>"

st.markdown(md, unsafe_allow_html=True)
