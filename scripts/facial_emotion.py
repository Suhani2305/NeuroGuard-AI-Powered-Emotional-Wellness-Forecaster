import cv2
import mediapipe as mp
import numpy as np

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
    # Example: Use mouth aspect ratio for smile detection
    if landmarks is None or len(landmarks) < 468:
        return 'No Face', 0.0
    # Mouth corners: 61 (left), 291 (right), 13 (top), 14 (bottom)
    left = np.array(landmarks[61])
    right = np.array(landmarks[291])
    top = np.array(landmarks[13])
    bottom = np.array(landmarks[14])
    mouth_width = np.linalg.norm(left - right)
    mouth_height = np.linalg.norm(top - bottom)
    ratio = mouth_height / (mouth_width + 1e-6)
    if ratio > 0.38:
        return 'Surprised', ratio
    elif ratio > 0.28:
        return 'Happy', ratio
    elif ratio < 0.18:
        return 'Neutral', ratio
    else:
        return 'Sad', ratio

def draw_landmarks(image, landmarks):
    for pt in landmarks:
        cv2.circle(image, pt, 1, (0,255,0), -1)

if __name__ == "__main__":
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        landmarks = get_face_landmarks(frame, mp_face_mesh)
        if landmarks:
            draw_landmarks(frame, landmarks)
            emotion, score = classify_emotion(landmarks)
            cv2.putText(frame, f"Emotion: {emotion} ({score:.2f})", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            cv2.putText(frame, "No Face Detected", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Facial Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows() 