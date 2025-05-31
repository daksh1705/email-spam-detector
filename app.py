import streamlit as st
import pickle
import numpy as np

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# App title
st.title("📧 Email Spam Classifier")

# User input
email = st.text_area("Paste your email content here:")

if st.button("Check Spam"):
    if email.strip() == "":
        st.warning("Please enter some email text.")
    else:
        # Transform input text
        email_vector = vectorizer.transform([email])
        
        # Predict
        prediction = model.predict(email_vector)[0]
        prob = model.predict_proba(email_vector)[0][1]  # Spam probability

        # Output
        if prediction == 1:
            st.error(f"🚨 This looks like **SPAM** ({prob*100:.2f}% confidence)")
        else:
            st.success(f"✅ This looks like **NOT SPAM** ({(1-prob)*100:.2f}% confidence)")
