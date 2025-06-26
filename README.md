# 🛡️ Fake Review Detector — ML & FastAPI-Based Web API

This project implements a machine learning–powered web service that classifies user reviews as **FAKE** or **GENUINE**, specifically for e-commerce and pharmaceutical platforms.

---

## 🎯 Objective

To design and deploy an end-to-end system that detects **fraudulent or paid reviews**, often seen on product websites, using:
- Natural Language Processing (NLP)
- Machine Learning (ML) ensemble techniques
- FastAPI for API creation
- Cloud deployment with automatic model loading

---

## 🔍 Problem Statement

Many online platforms suffer from fake reviews — either overly positive sponsored reviews or spammy negative ones. This project aims to **automate the detection of such reviews** using text-based features and a voting ensemble of classifiers.

---

## ✅ Key Features

- **Text Preprocessing**: Tokenization, stopword removal, punctuation stripping, stemming & lemmatization
- **TF-IDF Vectorization**: Converts textual data into weighted word vectors
- **Voting Classifier**:
  - Logistic Regression  
  - Random Forest  
  - Multinomial Naive Bayes  
- **Fake Detection Thresholding**: Reviews are flagged if fake probability > 0.6
- **REST API**: Built using FastAPI with `/predict` route
- **Google Drive Integration**: ML model is auto-downloaded from Google Drive if not present
- **Cross-Origin Support**: Configured for browser extension integration

---

## 🧠 Dataset & Model

- **Size**: 10,000 reviews (50% real, 50% fake)
- **Source**: Synthetic + real reviews scraped from pharmacy/e-commerce platforms
- **Fake Reviews**: Includes exaggerated positives and subtly paid content for robust training
- **Model Persistence**: Trained pipeline saved using `joblib` with protocol 4

---

## ⚙️ API Specification

**Endpoint**:
```http
POST /predict
Content-Type: application/json
{
  "review": "This medicine changed my life. Absolutely amazing!"
}
{
  "prediction": "🔴 FAKE Review",
  "fake_probability": 0.8421
}

# Clone the repo
git clone https://github.com/your-username/fake-review-detector-extension.git
cd fake-review-detector-extension

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```
# Run the server
python main.py

 Academic Highlights
Full-stack ML system: From preprocessing to deployment

API-first approach: Suitable for frontend integration (browser extension, React, etc.)

Clear abstraction: ML model logic separated from web interface

Efficient loading: Model file size optimized and downloaded only when needed

Author
Name: Om Srivastava
Institution: Netaji Subhas University Of Technology
Email: srivastavaom2207@gmail.com
GitHub: github.com/SrivXom


