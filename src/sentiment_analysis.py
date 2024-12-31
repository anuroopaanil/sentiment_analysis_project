# sentiment_analysis.py

from preprocessing import preprocess_text
from model import train_model, save_model
from evaluate import evaluate_model
import pandas as pd

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    data = pd.read_csv(file_path)
    return data

def main():
    # Load the dataset
    data = load_data("../data/IMDB Dataset.csv")

    # Preprocess the text data
    print("Preprocessing the data...")
    processed_text = preprocess_text(data["Review"])

    # Train a sentiment analysis model
    print("Training the model...")
    model = train_model(processed_text, data["Sentiment"])

    # Save the model
    print("Saving the model...")
    save_model(model)

    # Evaluate the model
    print("Evaluating the model...")
    accuracy = evaluate_model(model, processed_text, data["Sentiment"])
    print(f"Model Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
