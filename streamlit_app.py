import streamlit as st
import librosa
import numpy as np
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

# Load model
model = joblib.load("emotion_model.pkl")

# Emotion label map
emotion_dict = {
    1: "Neutral", 2: "Calm", 3: "Happy", 4: "Sad",
    5: "Angry", 6: "Fearful", 7: "Disgust", 8: "Surprised"
}

# CSV to store prediction history
HISTORY_FILE = "emotion_history.csv"

# Ensure history file exists
if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=["file", "emotion"]).to_csv(HISTORY_FILE, index=False)

def extract_features(file_path):
    y, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# App UI
st.title("🎙️ Advanced Speech Emotion Recognition")
st.markdown("Upload a `.wav` file and get emotion predictions with visualizations.")

# Upload
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"], accept_multiple_files=False)

if uploaded_file:
    # Save uploaded file
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f" File uploaded: {uploaded_file.name}")
    st.audio(uploaded_file, format="audio/wav")

    # Feature Extraction
    features = extract_features(file_path).reshape(1, -1)

    # Get Prediction Probabilities
    proba = model.predict_proba(features)[0]
    emotion_names = [emotion_dict[i+1] for i in range(len(proba))]

    # Display Probabilities
    st.subheader(" Emotion Probabilities")
    for i, p in enumerate(proba):
        st.write(f"{emotion_names[i]}: {p*100:.2f}%")

    # Bar Chart
    df_plot = pd.DataFrame({'Emotion': emotion_names, 'Probability': proba})
    st.bar_chart(df_plot.set_index("Emotion"))

    # Predicted emotion
    predicted_index = np.argmax(proba)
    predicted_emotion = emotion_names[predicted_index]
    st.success(f" Predicted Emotion: {predicted_emotion}")

    # Save to history
    history = pd.read_csv(HISTORY_FILE)
    new_entry = pd.DataFrame([[uploaded_file.name, predicted_emotion]], columns=["file", "emotion"])
    history = pd.concat([history, new_entry], ignore_index=True)
    history.to_csv(HISTORY_FILE, index=False)

    # Emotion Frequency Plot
    st.subheader(" Emotion Frequency (History)")
    counts = history["emotion"].value_counts()
    st.bar_chart(counts)

    # Predict next likely emotion
    st.subheader(" Future Emotion Prediction")
    emotion_sequence = history["emotion"].values[-5:]  # last 5 emotions
    if len(emotion_sequence) > 0:
        most_common = Counter(emotion_sequence).most_common(1)[0][0]
        st.info(f" Based on recent patterns, the next likely emotion is **{most_common}**.")
