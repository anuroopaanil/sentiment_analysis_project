import matplotlib.pyplot as plt
import pandas as pd

def visualize_sentiment_distribution():
    # Load the dataset
    data = pd.read_csv("data/IMDB Dataset.csv")

    # Visualize the sentiment distribution
    sentiment_counts = data['sentiment'].value_counts()
    sentiment_counts.plot(kind='bar', color=['green', 'red'])
    plt.title("Sentiment Distribution")
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    plt.show()
