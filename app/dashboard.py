import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime

st.set_page_config(page_title="NeuroGuard Dashboard", layout="wide")
st.title("🧠 NeuroGuard: Emotional Wellness Forecaster")
st.markdown("> Not just detecting how you feel — predicting how you will feel.")

# Tabs for input types
tabs = st.tabs(["Text Journal", "Voice Input", "Facial Analysis", "Mood Trend", "Suggestions"])

# --- Text Journal ---
with tabs[0]:
    st.header("📝 Text Journal")
    user_text = st.text_area("How are you feeling today? Write your thoughts:")
    if st.button("Analyze Text Emotion"):
        st.info("[Demo] Text emotion analysis would run here.")
        st.write("Emotion: Happy (0.92)")

# --- Voice Input ---
with tabs[1]:
    st.header("🎤 Voice Emotion")
    audio_file = st.file_uploader("Upload a voice recording (.wav)", type=["wav"])
    if st.button("Analyze Voice Emotion"):
        st.info("[Demo] Voice emotion analysis would run here.")
        st.write("Emotion: Neutral (0.65)")

# --- Facial Analysis ---
with tabs[2]:
    st.header("📷 Facial Emotion")
    st.info("[Demo] Webcam facial emotion detection would run here.")
    st.write("Emotion: Happy (0.78)")

# --- Mood Trend ---
with tabs[3]:
    st.header("📊 Mood Trendline")
    # Demo mood data
    dates = pd.date_range(datetime.date.today() - datetime.timedelta(days=14), periods=15)
    scores = [60, 62, 65, 70, 68, 72, 75, 78, 80, 77, 74, 70, 68, 65, 67]
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(dates, scores, marker='o', label='Mood Score')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mood Score')
    ax.set_title('Mood Trend (Demo)')
    ax.legend()
    st.pyplot(fig)

# --- Suggestions ---
with tabs[4]:
    st.header("🎁 Suggestions for You")
    st.write("- Listen to your favorite music 🎵")
    st.write("- Take a short walk 🚶‍♂️")
    st.write("- Try a 5-min breathing exercise 🧘‍♂️")
    st.write("- Inspirational quote: *'This too shall pass.'*")

# --- Privacy Note ---
st.markdown("---")
st.info("🔐 **Privacy:** All analysis runs locally. No data is stored or sent to the cloud.") 