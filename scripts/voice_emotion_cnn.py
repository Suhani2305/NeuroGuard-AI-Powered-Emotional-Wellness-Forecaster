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
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch

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

# Augmentation pipeline (reduced probability)
AUGMENT = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.2),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.2),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2)
])

# Feature extraction: MFCC only (for debugging)
def extract_features(file_path, max_pad_len=174, augment=False):
    y, sr = librosa.load(file_path, sr=None)
    if augment:
        y = AUGMENT(samples=y, sample_rate=sr)
    # MFCC only for now
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.pad(mfcc, ((0,0),(0,max(0,max_pad_len-mfcc.shape[1]))), mode='constant')[:, :max_pad_len]
    features = mfcc[..., np.newaxis]  # shape: (40, max_pad_len, 1)
    return features

# Load data with option to only augment minority classes
def load_data(data_dir, augment=False, augment_minority_only=False, class_counts=None, min_class_count=None):
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
            # Only augment if augment_minority_only and this is a minority class
            do_augment = augment
            if augment_minority_only and class_counts and min_class_count:
                if class_counts[emotion] > min_class_count:
                    do_augment = False
            feat = extract_features(file_path, augment=do_augment)
            X.append(feat)
            y.append(emotion)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    data_dir = 'data/voice'
    print('Loading data (no augmentation)...')
    X_full, y_full = load_data(data_dir, augment=False)
    print('Feature shape:', X_full.shape)
    print('Full class distribution:', Counter(y_full))
    le = LabelEncoder()
    y_enc = le.fit_transform(y_full)
    y_cat = to_categorical(y_enc)
    X_train, X_test, y_train, y_test, y_enc_train, y_enc_test = train_test_split(
        X_full, y_cat, y_enc, test_size=0.2, random_state=42, stratify=y_cat)
    # Print train/test class distribution
    print('Train class distribution:', Counter(le.inverse_transform(np.argmax(y_train, axis=1))))
    print('Test class distribution:', Counter(le.inverse_transform(np.argmax(y_test, axis=1))))

    # Augment only minority classes in training set
    train_labels = le.inverse_transform(np.argmax(y_train, axis=1))
    train_counts = Counter(train_labels)
    min_class_count = min(train_counts.values())
    print('Augmenting only minority classes in training set...')
    X_train_aug, y_train_aug = load_data(
        data_dir, augment=True, augment_minority_only=True, class_counts=train_counts, min_class_count=min_class_count)
    # Only keep augmented samples for training set indices
    X_train_aug = X_train_aug[:len(X_train)]
    y_train_aug = y_train[:len(X_train)]
    # Concatenate original and augmented
    X_train_final = np.concatenate([X_train, X_train_aug], axis=0)
    y_train_final = np.concatenate([y_train, y_train_aug], axis=0)

    num_classes = y_cat.shape[1]
    print('Building improved model...')
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(40, 174, 1)),
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

    # Compute class weights for imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_enc_train), y=y_enc_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print('Class weights:', class_weight_dict)

    print('Training...')
    model.fit(X_train_final, y_train_final, epochs=30, batch_size=32, validation_data=(X_test, y_test), class_weight=class_weight_dict)
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