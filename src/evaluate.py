# evaluate_model.py

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the trained model
model = joblib.load('sentiment_model.pkl')

# Load data (import from load_data.py)
from load_data import data

# Vectorize the test data
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_test_tfidf = vectorizer.transform(data['Overview'])

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print(f'Accuracy: {accuracy_score(data['Sentiment'], y_pred)}')
print(f'Classification Report:\n{classification_report(data['Sentiment'], y_pred)}')
print(f'Confusion Matrix:\n{confusion_matrix(data['Sentiment'], y_pred)}')
