# import pandas as pd

# # Load the dataset
# data = pd.read_csv(r'C:\Users\lenovo\OneDrive\Desktop\sentiment_analysis_project\src\data\IMDB_Dataset.csv')

# # Print the first few rows of the dataset to check the columns
# print(data.head())









import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load the dataset
data = pd.read_csv(r'C:\Users\lenovo\OneDrive\Desktop\sentiment_analysis_project\src\data\IMDB_Dataset.csv')

# Preprocessing
data['review'] = data['review'].str.replace(r'<.*?>', '', regex=True)  # Remove HTML tags
data['review'] = data['review'].str.replace(r'[^a-zA-Z\s]', '', regex=True)  # Remove non-alphabetic characters
data['review'] = data['review'].str.lower()  # Convert to lowercase

# Encode labels
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# Split the data
X = data['review']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text data
vectorizer = CountVectorizer(max_features=5000)
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

# Save the model and vectorizer
with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
