import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Emotion mapping from RAVDESS filename
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Feature extraction: MFCC + Chroma + MelSpectrogram
def extract_features(file_path, max_pad_len=174):
    y, sr = librosa.load(file_path, sr=None)
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc = np.pad(mfcc, ((0,0),(0,max(0,max_pad_len-mfcc.shape[1]))), mode='constant')[:, :max_pad_len]
    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma = np.pad(chroma, ((0,0),(0,max(0,max_pad_len-chroma.shape[1]))), mode='constant')[:, :max_pad_len]
    # MelSpectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel = np.pad(mel, ((0,0),(0,max(0,max_pad_len-mel.shape[1]))), mode='constant')[:, :max_pad_len]
    # Concatenate features along frequency axis
    features = np.concatenate([mfcc, chroma, mel], axis=0)  # shape: (160, max_pad_len)
    features = features[..., np.newaxis]  # shape: (160, max_pad_len, 1)
    return features

def load_data(data_dir):
    X, y = [], []
    for actor in os.listdir(data_dir):
        actor_dir = os.path.join(data_dir, actor)
        if not os.path.isdir(actor_dir):
            continue
        for fname in tqdm(os.listdir(actor_dir), desc=actor):
            if not fname.endswith('.wav'):
                continue
            parts = fname.split('-')
            if len(parts) < 3:
                continue
            emotion = emotion_map.get(parts[2], None)
            if emotion is None:
                continue
            file_path = os.path.join(actor_dir, fname)
            feat = extract_features(file_path)
            X.append(feat)
            y.append(emotion)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    data_dir = 'data/voice'
    print('Loading data...')
    X, y = load_data(data_dir)
    print('Feature shape:', X.shape)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_cat)

    num_classes = y_cat.shape[1]
    print('Building improved model...')
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(160, 174, 1)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.3),
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.3),
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.3),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print('Training...')
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc*100:.2f}%")
    model.save('models/voice_emotion_cnn.h5')

    # Confusion matrix and classification report
    y_pred = model.predict(X_test)
    y_pred_labels = le.inverse_transform(np.argmax(y_pred, axis=1))
    y_true_labels = le.inverse_transform(np.argmax(y_test, axis=1))
    print(classification_report(y_true_labels, y_pred_labels))
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=le.classes_)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show() 