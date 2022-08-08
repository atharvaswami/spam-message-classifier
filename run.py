import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
#nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import Parallel, delayed
import joblib
import pickle
import os.path

def clean_data(message):
    message_without_punc = [character for character in message if character not in string.punctuation]
    message_without_punc = ''.join(message_without_punc)

    separator = ' '
    return separator.join([word for word in message_without_punc.split() if word.lower() not in stopwords.words('english')])

def predict(text):
    labels = ['Not Spam', 'Spam']
    x = cv.transform(text).toarray()
    p = model.predict(x)
    s = [str(i) for i in p]
    v = int(''.join(s))
    return str('This message is: '+labels[v])

if __name__ == '__main__':
    df = pd.read_csv('spam.csv', encoding='Windows-1252')
    df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    df.rename(columns={'v1':'labels', 'v2':'message'}, inplace=True)
    df.drop_duplicates(inplace=True)

    df['labels'] = df['labels'].map({'ham':0, 'spam':1})

    df['message'] = df['message'].apply(clean_data)

    X = df['message']
    y = df['labels']

    cv = CountVectorizer()
    X = cv.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = MultinomialNB().fit(X_train, y_train)   

    st.title('Spam Classifier')
    st.image('spam.jpg')
    user_input = st.text_input('Write your message: ')
    submit = st.button('Predict')
    if submit:
        answer = predict([user_input])
        st.text(answer)
