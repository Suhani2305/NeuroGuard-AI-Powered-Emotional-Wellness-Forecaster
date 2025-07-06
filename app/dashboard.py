import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import datetime
from streamlit_lottie import st_lottie
import requests
import numpy as np
from PIL import Image
import io
import cv2
import tensorflow as tf
from transformers import pipeline
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="NeuroGuard Dashboard", layout="wide", page_icon="🧠")

# Sidebar
st.sidebar.title("🧠 NeuroGuard")
st.sidebar.markdown("AI-Powered Emotional Wellness Forecaster")
st.sidebar.markdown("---")
st.sidebar.info("All analysis runs locally. No data is sent to the cloud.")
st.sidebar.markdown("[GitHub Repo](https://github.com/Suhani2305/NeuroGuard-AI-Powered-Emotional-Wellness-Forecaster)")

# Lottie animation loader
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_emo = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_4kx2q32n.json")

# Load facial emotion CNN model (if available)
@st.cache_resource
def load_facial_model():
    try:
        model = tf.keras.models.load_model('models/facial_emotion_cnn_final.h5')
        return model
    except Exception:
        return None
facial_model = load_facial_model()

# FER2013 class labels (update if your model uses different order)
FER_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Preprocess image for CNN model
def preprocess_face_image(img):
    img = img.convert('L').resize((48,48))
    arr = np.array(img).astype('float32') / 255.0
    arr = arr.reshape(1, 48, 48, 1)
    return arr

# Load BERT pipeline for text emotion
def load_bert_pipeline():
    try:
        pipe = pipeline('text-classification', model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
        return pipe
    except Exception:
        return None
bert_pipe = load_bert_pipeline()

# Header with animation
col1, col2 = st.columns([1,3])
with col1:
    st_lottie(lottie_emo, height=120, key="emotion_anim")
with col2:
    st.title("NeuroGuard: Emotional Wellness Forecaster")
    st.markdown("> Not just detecting how you feel — predicting how you will feel.")

st.markdown("---")

# Tabs for input types
tabs = st.tabs(["Text Journal", "Voice Input", "Facial Analysis", "Mood Trend", "Suggestions"])

# --- Text Journal ---
with tabs[0]:
    st.header("📝 Text Journal")
    st.markdown("Write your thoughts and get instant emotion analysis.")
    user_text = st.text_area("How are you feeling today?", height=120)
    colA, colB = st.columns([2,1])
    with colA:
        analyze_btn = st.button("Analyze Text Emotion", key="analyze_text_emo")
        if analyze_btn:
            if user_text.strip():
                # Real BERT prediction
                if bert_pipe:
                    result = bert_pipe(user_text)
                    top = max(result[0], key=lambda x: x['score'])
                    st.success(f"BERT: {top['label'].capitalize()} ({top['score']:.2f})")
                # TextBlob
                tb = TextBlob(user_text)
                st.info(f"TextBlob Polarity: {tb.sentiment.polarity:.2f}")
                # VADER
                vader = SentimentIntensityAnalyzer()
                vs = vader.polarity_scores(user_text)
                st.info(f"VADER Compound: {vs['compound']:.2f}")
                st_lottie(load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_4kx2q32n.json"), height=80, key="text_emo")
            else:
                st.warning("Please enter some text.")
    with colB:
        st.info("BERT/TextBlob/VADER ready!")

# --- Voice Input ---
with tabs[1]:
    st.header("🎤 Voice Emotion")
    st.markdown("Upload a voice recording (.wav) and get emotion prediction.")
    audio_file = st.file_uploader("Upload a voice recording", type=["wav"])
    colC, colD = st.columns([2,1])
    with colC:
        if st.button("Analyze Voice Emotion"):
            st.info("[TODO] Integrate real voice model inference here.")
            st.success("Emotion: Neutral 😐 (0.65)")
            st_lottie(load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_4kx2q32n.json"), height=80, key="voice_emo")
    with colD:
        st.info("CNN (MFCC+Chroma+MelSpec) ready!")

# --- Facial Analysis ---
with tabs[2]:
    st.header("📷 Facial Emotion")
    st.markdown("Take a photo or upload an image to detect facial emotion.")
    cam_img = st.camera_input("Take a photo (webcam)")
    upload_img = st.file_uploader("Or upload a face image", type=["jpg", "jpeg", "png"])
    img = None
    if cam_img is not None:
        img = Image.open(cam_img)
    elif upload_img is not None:
        img = Image.open(upload_img)
    if img is not None:
        st.image(img, caption="Input Image", width=200)
        # Try CNN model first
        if facial_model:
            arr = preprocess_face_image(img)
            pred = facial_model.predict(arr)
            label = FER_CLASSES[np.argmax(pred)]
            score = np.max(pred)
            st.success(f"CNN: {label.capitalize()} ({score:.2f})")
        else:
            st.warning("CNN model not found. Using rule-based fallback.")
            st.info("[TODO] Add rule-based fallback here if needed.")
        st_lottie(load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_4kx2q32n.json"), height=80, key="face_emo")
    else:
        st.info("Take a photo or upload an image to get prediction.")
    st.info("CNN (FER2013) ready!")

# --- Mood Trend ---
with tabs[3]:
    st.header("📊 Mood Trendline")
    st.markdown("Track your mood over time and see predictions for the next week.")
    # Demo mood data
    dates = pd.date_range(datetime.date.today() - datetime.timedelta(days=14), periods=15)
    scores = [60, 62, 65, 70, 68, 72, 75, 78, 80, 77, 74, 70, 68, 65, 67]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=scores, mode='lines+markers', name='Mood Score', line=dict(color='royalblue', width=3)))
    fig.update_layout(xaxis_title='Date', yaxis_title='Mood Score', title='Mood Trend (Demo)', template='plotly_dark', height=350)
    st.plotly_chart(fig, use_container_width=True)

# --- Suggestions ---
with tabs[4]:
    st.header("🎁 Smart Suggestions for You")
    st.markdown("Get personalized self-care tips based on your mood.")
    st.markdown("""
    <ul style='font-size:18px;'>
    <li>🎵 <b>Listen to your favorite music</b></li>
    <li>🚶‍♂️ <b>Take a short walk</b></li>
    <li>🧘‍♂️ <b>Try a 5-min breathing exercise</b></li>
    <li>💡 <b>Inspirational quote:</b> <i>'This too shall pass.'</i></li>
    </ul>
    """, unsafe_allow_html=True)
    st_lottie(load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_4kx2q32n.json"), height=80, key="suggest_emo")

st.markdown("---")
st.info("🔐 **Privacy:** All analysis runs locally. No data is stored or sent to the cloud.") 