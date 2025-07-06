import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

def get_face_landmarks(image, face_mesh):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        coords = [(int(pt.x * image.shape[1]), int(pt.y * image.shape[0])) for pt in landmarks.landmark]
        return coords
    return None

# Simple rule-based emotion classifier (smile detection)
def classify_emotion(landmarks):
    if landmarks is None or len(landmarks) < 468:
        return 'No Face', 0.0
    left = np.array(landmarks[61])
    right = np.array(landmarks[291])
    top = np.array(landmarks[13])
    bottom = np.array(landmarks[14])
    mouth_width = np.linalg.norm(left - right)
    mouth_height = np.linalg.norm(top - bottom)
    ratio = mouth_height / (mouth_width + 1e-6)
    if ratio > 0.38:
        return 'surprise', ratio
    elif ratio > 0.28:
        return 'happy', ratio
    elif ratio < 0.18:
        return 'neutral', ratio
    else:
        return 'sad', ratio

def batch_evaluate(test_dir):
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    y_true, y_pred = [], []
    classes = sorted(os.listdir(test_dir))
    for label in classes:
        img_dir = os.path.join(test_dir, label)
        for fname in tqdm(os.listdir(img_dir), desc=label):
            if not fname.endswith('.jpg'):
                continue
            img_path = os.path.join(img_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            landmarks = get_face_landmarks(img, mp_face_mesh)
            pred, _ = classify_emotion(landmarks)
            y_true.append(label)
            y_pred.append(pred)
    acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {acc*100:.2f}% on {len(y_true)} images")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    # Batch evaluation on FER2013 test set
    batch_evaluate('data/face/test') 