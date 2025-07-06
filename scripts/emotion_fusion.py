import numpy as np
from collections import Counter

def normalize_score(score, min_score=0.0, max_score=1.0):
    # Normalize any score to 0-1
    return (score - min_score) / (max_score - min_score + 1e-6)

def fuse_emotions(emotion_results, weights=None):
    """
    emotion_results: list of dicts, e.g.
      [
        {'modality': 'text', 'label': 'Happy', 'score': 0.85},
        {'modality': 'voice', 'label': 'Neutral', 'score': 0.60},
        {'modality': 'face', 'label': 'Happy', 'score': 0.70}
      ]
    weights: dict, e.g. {'text': 0.4, 'voice': 0.3, 'face': 0.3}
    """
    if weights is None:
        weights = {'text': 0.4, 'voice': 0.3, 'face': 0.3}
    # Normalize scores and apply weights
    mood_score = 0.0
    label_votes = []
    for res in emotion_results:
        norm_score = normalize_score(res['score'])
        mood_score += norm_score * weights.get(res['modality'], 0.3)
        label_votes.append(res['label'])
    mood_score = int(mood_score * 100)  # 0-100
    # Dominant emotion by majority vote
    dominant = Counter(label_votes).most_common(1)[0][0]
    return mood_score, dominant

if __name__ == "__main__":
    # Demo: sample results from each modality
    sample_results = [
        {'modality': 'text', 'label': 'Happy', 'score': 0.92},
        {'modality': 'voice', 'label': 'Neutral', 'score': 0.65},
        {'modality': 'face', 'label': 'Happy', 'score': 0.78}
    ]
    mood_score, dominant = fuse_emotions(sample_results)
    print(f"Unified Mood Score: {mood_score}/100")
    print(f"Dominant Emotion: {dominant}") 