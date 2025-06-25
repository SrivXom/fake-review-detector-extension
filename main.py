from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, string, nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize
import uvicorn

# ⬇️ FIRST: Create the FastAPI app
app = FastAPI()

# ⬇️ THEN: Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download NLTK data (only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize preprocessing tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words and not word.isdigit()]
    stemmed = [stemmer.stem(word) for word in tokens]
    lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]
    return ' '.join(lemmatized)

# Load model
model = joblib.load("tfidf_voting_model.pkl")

class ReviewRequest(BaseModel):
    review: str

@app.post("/predict")
def predict_review(request: ReviewRequest):
    cleaned = clean_text(request.review)
    probs = model.predict_proba([cleaned])[0]
    class_idx = model.classes_.tolist().index('FAKE')
    fake_prob = probs[class_idx]

    return {
        "prediction": "🔴 FAKE Review" if fake_prob > 0.6 else "🟢 GENUINE Review",
        "fake_probability": round(fake_prob, 4)
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
