import re
import torch
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Baseline sentiment using TextBlob
def textblob_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Baseline sentiment using VADER
def vader_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    return score['compound']

# BERT-based emotion classification (using HuggingFace pipeline)
bert_classifier = pipeline('text-classification', model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)

def bert_emotion(text):
    results = bert_classifier(text)
    # Get top emotion
    top = max(results[0], key=lambda x: x['score'])
    return top['label'], top['score']

if __name__ == "__main__":
    sample_text = "I am feeling very happy and excited today!"
    clean_text = preprocess_text(sample_text)
    print(f"TextBlob Sentiment: {textblob_sentiment(clean_text)}")
    print(f"VADER Sentiment: {vader_sentiment(clean_text)}")
    emotion, score = bert_emotion(clean_text)
    print(f"BERT Emotion: {emotion} (score: {score:.2f})") 