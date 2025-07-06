import os
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    features = np.hstack([mfccs, chroma, zcr])
    return features

# Train SVM model (demo, expects data in X, y)
def train_svm(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42)
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    return clf, scaler, le

# Predict emotion from audio file
def predict_emotion(file_path, clf, scaler, le):
    features = extract_features(file_path).reshape(1, -1)
    features_scaled = scaler.transform(features)
    pred = clf.predict(features_scaled)
    prob = clf.predict_proba(features_scaled)
    return le.inverse_transform(pred)[0], np.max(prob)

if __name__ == "__main__":
    # Demo: placeholder for loading data and training
    # Example usage: Place some .wav files in 'data/voice/' with filenames as 'emotion_something.wav'
    data_dir = 'data/voice/'
    X, y = [], []
    if os.path.exists(data_dir):
        for fname in os.listdir(data_dir):
            if fname.endswith('.wav'):
                label = fname.split('_')[0]  # expects 'emotion_xxx.wav'
                feat = extract_features(os.path.join(data_dir, fname))
                X.append(feat)
                y.append(label)
        if X:
            clf, scaler, le = train_svm(np.array(X), np.array(y))
            joblib.dump((clf, scaler, le), 'models/voice_svm.pkl')
            print('Model trained and saved!')
    # Demo prediction (if model exists)
    model_path = 'models/voice_svm.pkl'
    if os.path.exists(model_path):
        clf, scaler, le = joblib.load(model_path)
        test_file = 'data/voice/happy_test.wav'  # replace with your test file
        if os.path.exists(test_file):
            emotion, score = predict_emotion(test_file, clf, scaler, le)
            print(f'Predicted Emotion: {emotion} (confidence: {score:.2f})')
        else:
            print('Test audio file not found.')
    else:
        print('No trained model found. Please add data and retrain.') 