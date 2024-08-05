import pandas as pd
import numpy as np
import sklearn 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import streamlit as st

# Load and preprocess data
data = pd.read_csv(r"C:\Users\sharm\Downloads\spam mail.csv")
data['category'] = data['Category'].map({'spam': 1, 'ham': 0})
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(data['Masseges'])
y = data['category']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Train model
model = SVC()
model.fit(x_train, y_train)

# Evaluate model
score = model.score(x_test, y_test)
print("Model accuracy:", score)

# Streamlit app
st.markdown("<h1>Email Spam Detector </h1>",unsafe_allow_html=True)

st.markdown("<h3> Check whether your Mail  is Spam or Ham:</h3>",unsafe_allow_html=True)
st.image(r"C:\Users\sharm\Downloads\spamm.jpg",width=400)
st.markdown("<h3> Please enter the Email </h3>",unsafe_allow_html=True)
email = st.text_input("Drop your Mail here")
if st.button("Check"):
    if email:
        new_mail = vectorizer.transform([email])
        prediction = model.predict(new_mail)
        label_map = {1: 'spam', 0: 'ham'}
        label = label_map[prediction[0]]
        if label=='spam':
            st.error("Goshhh!!! The Email is Spam")
        else:
            st.success("Hurrayy!!! The Email is Ham")
    else:
        st.error("Please enter an email")
