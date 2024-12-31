import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pickle

# Load the dataset
data = pd.read_csv(r'C:\Users\lenovo\OneDrive\Desktop\sentiment_analysis_project\src\data\IMDB_Dataset.csv')

# Check the dataset columns
print(data.columns)

# Rename columns for consistency (if necessary)
if 'review' not in data.columns or 'sentiment' not in data.columns:
    raise ValueError("Dataset must have 'review' and 'sentiment' columns.")

# Encode the sentiment column (positive=1, negative=0)
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# Split data into features and target
X = data['review']
y = data['sentiment']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to numerical vectors using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Evaluate the model
y_pred = model.predict(X_test_vectorized)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and vectorizer
with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer have been saved successfully!")
