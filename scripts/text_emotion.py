import re
import torch
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import pandas as pd
from sklearn.metrics import accuracy_score

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

# Batch evaluation for dataset
def evaluate_on_file(file_path, max_samples=None):
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            if ';' not in line:
                continue
            text, label = line.strip().split(';')
            texts.append(preprocess_text(text))
            labels.append(label)
    # BERT predictions
    bert_preds = []
    for t in texts:
        pred, _ = bert_emotion(t)
        bert_preds.append(pred.lower())
    acc_bert = accuracy_score(labels, bert_preds)
    print(f"BERT Accuracy: {acc_bert*100:.2f}% on {len(labels)} samples")
    # TextBlob/VADER are sentiment, not emotion, so we skip accuracy for them here

if __name__ == "__main__":
    # Demo: batch evaluation on test.txt (change to train.txt for full train set)
    print("Evaluating on test set...")
    evaluate_on_file('data/text/test.txt', max_samples=200)  # Remove max_samples for full set 