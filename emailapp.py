import pickle
import string

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(content):
    content = content.lower()
    content = nltk.word_tokenize(content)

    y = []
    for i in content:
        if i.isalnum():
            y.append(i)

    content = y[:]
    y.clear()

    for i in content:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    content = y[:]
    y.clear()

    for i in content:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Email Classifier')
input_text = st.text_input("Enter the message")

if st.button("Predict"):

    transformed_text = transform_text(input_text)

    vector_input = tfidf.transform([transformed_text])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("NON ABUSIVE")
    else:
        st.header("ABUSIVE")