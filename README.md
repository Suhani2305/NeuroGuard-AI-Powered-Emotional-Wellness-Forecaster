# NeuroGuard: AI-Powered Emotional Wellness Forecaster

> “Not just detecting how you feel — predicting how you will feel.”

## Project Overview
NeuroGuard is a multi-modal emotional health prediction system using voice, text, and facial expression inputs. It detects your current emotional state and forecasts mood trends using advanced AI/ML techniques.

## Folder Structure
```
NeuroGuard/
├── data/
├── models/
├── app/
├── scripts/
├── README.md
```

## Roadmap
- Text, Voice, and Facial Emotion Analysis
- Multi-modal Emotion Fusion
- Mood Forecasting
- Streamlit Dashboard
- Deployment & Privacy

## Getting Started

### 1. Clone the Repository
```bash
# (If not already cloned)
git clone https://github.com/Suhani2305/NeuroGuard-AI-Powered-Emotional-Wellness-Forecaster.git
cd NeuroGuard-AI-Powered-Emotional-Wellness-Forecaster
```

### 2. Create and Activate Virtual Environment (Windows)
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run app/dashboard.py
```

---

## Troubleshooting
- If you get errors for missing packages, ensure your virtual environment is activated and run `pip install -r requirements.txt` again.
- For webcam/mic features, allow permissions when prompted.
- For Prophet install issues, try: `pip install pystan==2.19.1.1 prophet` (or use fbprophet as in requirements).

---

## Privacy Note
All analysis runs locally. No data is stored or sent to the cloud.

## Facial Emotion Recognition (CNN - Keras/TensorFlow)

For real-world accuracy, use the provided CNN model:

### 1. Install TensorFlow (if not already):
```bash
pip install tensorflow
```

### 2. Train the CNN on FER2013:
```bash
python scripts/facial_emotion_cnn.py
```
- This will train a CNN on your `data/face/train` and evaluate on `data/face/test`.
- Best model will be saved as `models/facial_emotion_cnn.h5` and final as `models/facial_emotion_cnn_final.h5`.

### 3. Evaluate/Test:
- After training, the script will print test accuracy.
- You can load and use the model for inference in your app/dashboard.

---

## Voice Emotion Recognition (CNN - Keras/TensorFlow)

For real-world accuracy, use the provided CNN model for RAVDESS:

### 1. Install TensorFlow and librosa (if not already):
```bash
pip install tensorflow librosa tqdm
```

### 2. Train the CNN on RAVDESS:
```bash
python scripts/voice_emotion_cnn.py
```
- This will train a CNN on your `data/voice/Actor_XX/*.wav` files.
- Best model will be saved as `models/voice_emotion_cnn.h5`.

### 3. Evaluate/Test:
- After training, the script will print test accuracy.
- You can load and use the model for inference in your app/dashboard.

---

 