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

 