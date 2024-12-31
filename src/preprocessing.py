import re

# Function to preprocess text (clean the review)
def preprocess_text(text):
    text = re.sub(r'<br />', ' ', text)  # Remove <br /> tags
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    return text
import re

def preprocess_text(text):
    """Clean and preprocess the text data."""
    # Convert to lowercase
    text = text.lower()
    # Handle negations
    text = re.sub(r"\b(not good)\b", "not_good", text)
    text = re.sub(r"\b(very good)\b", "very_good", text)
    text = re.sub(r"\b(not excellent)\b", "not_excellent", text)
    # Remove special characters, numbers, and punctuations
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Strip extra spaces
    text = text.strip()
    return text
