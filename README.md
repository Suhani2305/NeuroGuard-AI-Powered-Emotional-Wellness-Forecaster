# NeuroGuard: AI-Powered Emotional Wellness Forecaster

> “Not just detecting how you feel — predicting how you will feel.”

---

## 🌟 Project Overview
NeuroGuard is a multi-modal AI system that analyzes **text**, **voice**, and **facial expressions** to detect your current emotional state and forecast your mood trends. Built for research, competition, and real-world use, it combines deep learning, NLP, and signal processing in a unified dashboard.

---

## 🚀 Features
- **Text Emotion Analysis** (BERT, TextBlob, VADER)
- **Voice Emotion Recognition** (MFCC+Chroma+MelSpectrogram + CNN)
- **Facial Emotion Recognition** (FER2013 + CNN)
- **Multi-modal Fusion** (ensemble/weighted)
- **Mood Forecasting** (Prophet)
- **Streamlit Dashboard** (live demo, journaling, suggestions)
- **Privacy-first** (runs locally, no data sent to cloud)

---

## 📁 Folder Structure
```
NeuroGuard/
├── data/         # Datasets (text, voice, face)
├── models/       # Trained models
├── app/          # Streamlit dashboard
├── scripts/      # All core modules
├── README.md
```

---

## ⚡ Quickstart

### 1. Clone & Setup
```bash
git clone https://github.com/Suhani2305/NeuroGuard-AI-Powered-Emotional-Wellness-Forecaster.git
cd NeuroGuard-AI-Powered-Emotional-Wellness-Forecaster
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the Dashboard
```bash
streamlit run app/dashboard.py
```

---

## 🧠 Module Details

### 1. Text Emotion Analysis
- **Script:** `scripts/text_emotion.py`
- **Model:** BERT (HuggingFace), TextBlob, VADER
- **Dataset:** [Kaggle Emotions Dataset](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)
- **How to run:**
  ```bash
  python scripts/text_emotion.py
  ```
  - Evaluates BERT on test set, prints accuracy.

### 2. Voice Emotion Recognition (CNN)
- **Script:** `scripts/voice_emotion_cnn.py`
- **Features:** MFCC + Chroma + MelSpectrogram
- **Model:** Deep CNN (Keras/TensorFlow)
- **Dataset:** [RAVDESS](https://zenodo.org/record/1188976)
- **How to run:**
  ```bash
  pip install tensorflow librosa tqdm matplotlib seaborn
  python scripts/voice_emotion_cnn.py
  ```
  - Trains and evaluates, prints accuracy and confusion matrix.

### 3. Facial Emotion Recognition (CNN)
- **Script:** `scripts/facial_emotion_cnn.py`
- **Model:** Deep CNN (Keras/TensorFlow)
- **Dataset:** [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
- **How to run:**
  ```bash
  pip install tensorflow
  python scripts/facial_emotion_cnn.py
  ```
  - Trains and evaluates, prints accuracy.

### 4. Multi-modal Fusion
- **Script:** `scripts/emotion_fusion.py`
- **How to run:**
  ```bash
  python scripts/emotion_fusion.py
  ```
  - Combines text, voice, face scores to unified mood score.

### 5. Mood Forecasting
- **Script:** `scripts/mood_forecast.py`
- **Model:** Prophet
- **How to run:**
  ```bash
  pip install prophet
  python scripts/mood_forecast.py
  ```
  - Predicts mood trend, plots graph.

### 6. Streamlit Dashboard
- **Script:** `app/dashboard.py`
- **How to run:**
  ```bash
  streamlit run app/dashboard.py
  ```
  - Unified UI for all features.

---

## 🛠️ Training & Customization
- Place your datasets in the `data/` folder as per the structure.
- Edit scripts for custom models, features, or datasets.
- Trained models are saved in `models/` and can be loaded for inference.

---

## 🐞 Troubleshooting
- **Missing packages:** Ensure venv is activated, run `pip install -r requirements.txt`.
- **Prophet install issues:** Try `pip install prophet`.
- **Webcam/mic:** Allow permissions in browser/OS.
- **GPU:** For faster training, use a machine with NVIDIA GPU and install TensorFlow GPU version.

---

## 🔐 Privacy
All analysis runs locally. No data is sent to the cloud. You own your data.

---

## 📚 References
- [RAVDESS Dataset](https://zenodo.org/record/1188976)
- [Kaggle Emotions Dataset](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)
- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

---

## 👑 Author & Credits
Developed by Suhani2305 (and contributors). For academic, research, and demo use.

---

 