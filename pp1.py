# stream lit code
pip install beautifulsoup4
pip install lxml

import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Function to fetch webpage content

import joblib
import numpy as np
from urllib.parse import urlparse
import re
import collections
from tld import get_tld

# Load the trained model, scaler, and label encoder
model = joblib.load('voting_model.pkl')
scaler = joblib.load('scaler.pkl')
lb = joblib.load('label_encoder.pkl')


# Feature extraction functions

def count_www(url):
    return url.count('www')

def count_https(url):
    return url.count('https')

def calculate_entropy(url):
    url = url.strip()
    prob = [float(url.count(c)) / len(url) for c in dict(collections.Counter(url))]
    entropy = - sum([p * np.log2(p) for p in prob])
    return entropy

def count_http(url):
    return url.count('http')

def count_hyphen(url):
    return url.count('-')

def url_length(url):
    return len(str(url))

def hostname_len(url):
    return len(urlparse(url).netloc)

def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                      url)
    if match:
        return 1
    else:
        return 0

def extract_tld(url):
    return get_tld(url, fail_silently=True)

# Define the function to calculate the length of the TLD
def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1

def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits


def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters


def extract_features(url):
    tld = extract_tld(url),
    return [

        count_www(url),
        count_https(url),
        calculate_entropy(url),
        count_http(url),
        count_hyphen(url),
        url_length(url),
        hostname_len(url),
        suspicious_words(url),
        tld_length(tld),
        digit_count(url),
        letter_count(url),

    ]

# Function to predict if a URL is spam
def predict_url_spam(url):
    features = extract_features(url)
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    predicted_class = lb.inverse_transform(prediction)
    return predicted_class[0]
def fetch_webpage_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            return None
    except Exception as e:
        st.error("Error fetching webpage content.")
        st.error(e)
        return None

# Function to classify URL based on content
def classify_url(url):
    webpage_content = fetch_webpage_content(url)
    if webpage_content:
        # Use BeautifulSoup to parse HTML content
        soup = BeautifulSoup(webpage_content, 'html.parser')
        # Extract text content from webpage
        text_content = soup.get_text()
        # Train a simple classifier (e.g., Multinomial Naive Bayes) using TF-IDF vectors of text content
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform([text_content])
        # Train a simple classifier (e.g., Multinomial Naive Bayes)
        clf = MultinomialNB()
        clf.fit(X, [0])  # Dummy label since we only have one sample
        # Predict class (0: legitimate, 1: phishing)
        prediction = clf.predict(X)[0]
        return "Legitimate" if prediction == 0 else "Phishing"
    else:
        return "Phishing"

# Streamlit UI
st.title("Phishing Website Classifier")

url = st.text_input("Enter URL:", "")
if url:
    result = predict_url_spam(url)
    if(result==0):
        classification = classify_url(url)
    else:

        classification = "Phishing"

    st.write(f"Classification for {url}: {classification}")

    if classification == "Phishing":
        st.error("This website might be a phishing website. Be cautious!")
    else:
        st.success("This website seems legitimate.")

