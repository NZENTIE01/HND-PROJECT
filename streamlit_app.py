import matplotlib.pyplot as plt
import librosa.display

import streamlit as st
import librosa
import librosa.display
import numpy as np
import joblib
import os
import pandas as pd
import plotly.graph_objects as go
from collections import Counter

# Emotion label map
emotion_dict = {
    1: "Neutral", 2: "Calm", 3: "Happy", 4: "Sad",
    5: "Angry", 6: "Fearful", 7: "Disgust", 8: "Surprised"
}

# Paths
HISTORY_FILE = "emotion_history.csv"
if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=["file", "emotion", "actor"]).to_csv(HISTORY_FILE, index=False)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("emotion_model.pkl")

model = load_model()

# Feature extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0), y, sr

# Guess actor ID from filename (assumes filenames like "03-01-01-01-01-01-12.wav" where 12 is actor)
def guess_actor_from_filename(filename):
    try:
        actor_id = filename.split("-")[-1].replace(".wav", "")
        return f"Actor_{int(actor_id):02d}"
    except:
        return "Unknown"

# UI
st.title("🎙️ Speech Emotion Recognition with Dynamic Waveform & Actor Detection")

# header
st.header("speech analysis")

# sub header
st.subheader("emotion recognition")

# markdown
st.markdown("Upload a `.wav` file to detect **emotion**, **visualize waveform**, and **guess actor identity**.")

# Upload section
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file:
    # Save file locally
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file, format="audio/wav")
        # Load audio for waveform
    y, sr = librosa.load(file_path, sr=None)

    # Display raw audio waveform using librosa + matplotlib
    st.subheader("📉 Raw Audio Waveform")

    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")
    st.pyplot(fig)


    # Extract features and audio
    features, signal, sr = extract_features(file_path)
    features = features.reshape(1, -1)

    # Predict emotion probabilities
    proba = model.predict_proba(features)[0]
    emotion_names = [emotion_dict[i+1] for i in range(len(proba))]

    # Display emotion percentages
    st.subheader("🎯 Emotion Prediction Probabilities")
    for i, p in enumerate(proba):
        st.write(f"{emotion_names[i]}: **{p*100:.2f}%**")

    # Emotion bar chart
    st.subheader("📊 Emotion Probability Chart")
    emotion_df = pd.DataFrame({'Emotion': emotion_names, 'Probability (%)': proba * 100})
    st.bar_chart(emotion_df.set_index("Emotion"))

    # 📉 Dynamic waveform
    st.subheader("🎧 Dynamic Raw Audio Waveform")
    time = np.linspace(0, len(signal)/sr, num=len(signal))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=signal, mode='lines', name='Waveform'))
    fig.update_layout(title="Interactive Audio Waveform", xaxis_title="Time (s)", yaxis_title="Amplitude")
    st.plotly_chart(fig)

    # Final emotion
    predicted_index = np.argmax(proba)
    predicted_emotion = emotion_names[predicted_index]
    st.success(f"✅ Predicted Emotion: **{predicted_emotion}**")

    # Guess actor from filename
    actor_guess = guess_actor_from_filename(uploaded_file.name)
    st.info(f"🧑‍🎤 Detected Actor ID (from file): **{actor_guess}**")

    # Save to history
    history = pd.read_csv(HISTORY_FILE)
    new_row = pd.DataFrame([[uploaded_file.name, predicted_emotion, actor_guess]],
                           columns=["file", "emotion", "actor"])
    history = pd.concat([history, new_row], ignore_index=True)
    history.to_csv(HISTORY_FILE, index=False)

    # Emotion frequency over time
    st.subheader("📈 Emotion Frequency (History)")
    counts = history["emotion"].value_counts()
    st.bar_chart(counts)

    # Predict next emotion based on history
    st.subheader("🔮 Predicted Next Likely Emotion")
    recent_emotions = history["emotion"].values[-5:]
    if len(recent_emotions) > 0:
        likely_next = Counter(recent_emotions).most_common(1)[0][0]
        st.info(f"Based on recent patterns, next likely emotion: **{likely_next}**")
