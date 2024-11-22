# Combined Feature-Based and Content-Based URL Analysis for Improved Phishing Detection
Phishing attacks pose a growing cybersecurity threat, targeting individuals and organizations to steal sensitive information. This project presents a hybrid approach that combines feature-based and content-based URL analysis to effectively detect and mitigate phishing threats with high accuracy and reduced false positives.

## 🚀 Overview
This project leverages machine learning models and ensemble techniques to analyze malicious URLs using two key methods:

Feature-Based Analysis: Extracts structural and statistical features of URLs (e.g., length, special characters, use of IP addresses).
Content-Based Analysis: Analyzes the textual content of the webpages, such as HTML and visible text.
The system is designed to operate in real-time, providing robust protection against phishing attacks across multiple domains.

## 🛠️ Features
Hybrid Detection: Combines feature-based and content-based analysis for better detection rates.

Ensemble Learning: Uses models like Random Forest, AdaBoost, and Gradient Boosting for high accuracy.

Real-Time Monitoring: Scans URLs continuously and provides automated alerts for detected phishing attempts.

Accuracy: Achieves 99.69% accuracy with minimized false positives.

Extensibility: Future enhancements include deep learning integration and cross-platform adoption.



## 📂 Repository Structure
├── 📁 .devcontainer         # Dev container configuration files for development environment

├── label_encoder.pkl        # Pre-trained label encoder for the model

├── pp1.py                   # Script to run the phishing detection system

├── requirements.txt         # Python dependencies for the project

├── scaler.pkl               # Pre-trained scaler for feature normalization

├── urldata.csv              # Dataset containing labeled URLs (benign and malicious)

├── urldata.csv:Zone.Identifier # Metadata for the dataset

├── voting_model.pkl         # Trained ensemble model for URL classification

├── voting_model.pkl:Zone.Identifier # Metadata for the model

├── x.py                    # Script for preprocessing and feature extraction


├── README.md                # Project overview and instructions


## 🛠️ Setup and Installation
  Clone the repository:
        
        git clone https://github.com/manasagangotri/Combined-Feature-Based-and-Content-Based-URL-Analysis-for-Improved-Phishing-Detection.git
        
        cd  Combined-Feature-Based-and-Content-Based-URL-Analysis-for-Improved-Phishing-Detection
  
  Install dependencies:
        
        pip install -r requirements.txt
  
  Run the phishing detection script:
        
        python3 pp1.py
## 🧪 How It Works
  
  Data Preprocessing: x.py handles feature extraction and normalization using scaler.pkl and label_encoder.pkl.
  
  URL Classification: The voting_model.pkl is a pre-trained ensemble model that combines several algorithms for high-accuracy URL classification.
  
  Input Format: Accepts URLs via urldata.csv or through real-time testing in the demo interface.
## 📊 Results

Hybrid Model Accuracy: 99.69%

Key Features:URL length, special character counts (?, %, etc.), and presence of IP addresses.

## 🌐 Applications
Real-Time Phishing Detection:Blocks malicious URLs before users access them.

Enterprise Security:Helps organizations secure networks by identifying phishing attempts.

Cybersecurity Education:Demonstrates how hybrid models improve phishing detection.
## 🚀 Future Enhancements

Integration of deep learning models for improved accuracy.

Real-time updates from live threat intelligence feeds.

Development of browser plugins for seamless phishing protection.

# 🚀 Demo

Experience the project in action:

👉 Click here to try the live demo! https://manasagangotri-combined-feature-based-and-content-ba-pp1-xnclqa.streamlit.app/

