import streamlit as st
import numpy as np
import librosa
import joblib
import os

# Load the trained model
model = joblib.load("emotion_model.pkl")

# Emotion labels from RAVDESS (map class index to emotion name)
emotion_dict = {
    1: "Neutral",
    2: "Calm",
    3: "Happy",
    4: "Sad",
    5: "Angry",
    6: "Fearful",
    7: "Disgust",
    8: "Surprised"
}

# Feature extraction function
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Streamlit UI
st.title(" Speech Emotion Recognition")
st.write("Upload a `.wav` file and the system will predict the speaker's emotion.")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(uploaded_file, format="audio/wav")
    st.success(" Audio uploaded successfully")

    try:
        features = extract_features("temp.wav")
        features = features.reshape(1, -1)
        prediction = model.predict(features)[0]
        emotion = emotion_dict.get(prediction, "Unknown")

        st.subheader(" Predicted Emotion:")
        st.markdown(f"###  {emotion}")

    except Exception as e:
        st.error(f" Error processing audio: {e}")

