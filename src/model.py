# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# import joblib
# from preprocessing import preprocess_text

# # Load your data
# data = pd.read_csv(r'C:\Users\lenovo\OneDrive\Desktop\sentiment_analysis_project\src\data\IMDB_Dataset.csv')

# # Print column names to check
# print(data.columns)

# # Apply preprocessing to the review column (change 'text' to 'review')
# data['review'] = data['review'].apply(preprocess_text)

# # Vectorize the text (make sure to use the correct vectorizer)
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(data['review'])

# # Train your model (ensure you've split data, for now using all)
# model = LogisticRegression()
# model.fit(X, data['sentiment'])

# # Save the model and vectorizer for later use
# joblib.dump(model, 'sentiment_model.pkl')
# joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# # Function for sentiment prediction
# def predict_sentiment(text):
#     processed_text = preprocess_text(text)
#     text_features = vectorizer.transform([processed_text])
#     prediction = model.predict(text_features)
    
#     # Convert numeric labels (e.g., 0 for negative, 1 for positive) to text labels
#     if prediction == 1:
#         return "positive"
#     else:
#         return "negative"






import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer

# Load the pre-trained model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def preprocess_text(text):
    # Basic preprocessing: remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

def predict_sentiment(review):
    processed_review = preprocess_text(review)
    review_vector = vectorizer.transform([processed_review])
    prediction = model.predict(review_vector)
    return "Positive" if prediction[0] == 1 else "Negative"


import pickle

# Load the trained model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def predict_sentiment(review):
    """Predict the sentiment of a given review."""
    review_vectorized = vectorizer.transform([review])  # Vectorize the input review
    prediction = model.predict(review_vectorized)  # Predict sentiment
    return "Positive" if prediction[0] == 1 else "Negative"



def predict_sentiment(review):
    """Predict the sentiment of a given review."""
    review_vectorized = vectorizer.transform([review])  # Vectorize the input review
    probabilities = model.predict_proba(review_vectorized)  # Get prediction probabilities
    prediction = model.predict(review_vectorized)  # Predict sentiment
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    confidence = max(probabilities[0])  # Confidence score
    return sentiment, confidence
