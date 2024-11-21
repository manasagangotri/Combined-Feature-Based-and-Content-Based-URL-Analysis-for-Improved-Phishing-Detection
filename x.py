import joblib
import numpy as np
from urllib.parse import urlparse
import re
import collections
from tld import get_tld
import pandas as pd

# Load the trained model, scaler, and label encoder
model = joblib.load('voting_model.pkl')
scaler = joblib.load('scaler.pkl')
lb = joblib.load('label_encoder.pkl')


# Feature extraction functions

def count_www(url):
    return url.count('www')

def count_https(url):
    return url.count('https')

def url_entropy(url):
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
    tld = extract_tld(url)
    return [

        count_www(url),
        count_https(url),
        url_entropy(url),
        count_http(url),
        count_hyphen(url),
        url_length(url),
        hostname_len(url),
        suspicious_words(url),
        tld_length(tld),
        digit_count(url),
        letter_count(url),

    ]
feature_names = [
    "count_www",
    "count_https",
    "url_entropy",
    "count_http",
    "count_hyphen",
    "url_length",
    "hostname_len",
    "suspicious_words",
    "tld_length",
    "digit_count",
    "letter_count",
]

# Function to predict if a URL is spam
def predict_url_spam(url):
    features = extract_features(url)
    features_df = pd.DataFrame([features], columns=feature_names)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    predicted_class = lb.inverse_transform(prediction)
    return predicted_class[0]

# Example URLs to test
test_urls = [
    'https://www.google.com',
    'https://www.netflix.com/in/',
    'https://www.amazon.com',
    'https://www.youtube.com',
    'https://www.facebook.com',
    'http://secure-login.bankofamerica.com.online-secure-login.com',
    'http://account.google.com.secure-login.auth.com',
    'http://paypal.com.user-verification.login-confirmation.com',
    'http://amazon.login.verification-secure.com',
    'http://facebook.com-login.account-security.com'
]

# Loop through the test URLs and print the prediction results
for url in test_urls:
    result = predict_url_spam(url)
    if  result==0:
        print(f"The URL '{url}' is classified as: Legitimate")
    else:
        print(f"The URL '{url}' is classified as: Malicious")
