import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# List of dataset directories
dataset_dirs = ["Actor_01", "Actor_02", "Actor_03", "Actor_04", "Actor_05", "Actor_06", "Actor_07", "Actor_08", "Actor_09", "Actor_10", "Actor_11", "Actor_12", "Actor_13", "Actor_14", "Actor_15", "Actor_16", "Actor_17", "Actor_18", "Actor_19", "Actor_20", "Actor_21", "Actor_22", "Actor_23", "Actor_24", "audio_speech_actors_01-24/Actor_01", "audio_speech_actors_01-24/Actor_02", "audio_speech_actors_01-24/Actor_03", "audio_speech_actors_01-24/Actor_04", "audio_speech_actors_01-24/Actor_05", "audio_speech_actors_01-24/Actor_06", "audio_speech_actors_01-24/Actor_07", "audio_speech_actors_01-24/Actor_08", "audio_speech_actors_01-24/Actor_09", "audio_speech_actors_01-24/Actor_10", "audio_speech_actors_01-24/Actor_11", "audio_speech_actors_01-24/Actor_12", "audio_speech_actors_01-24/Actor_13", "audio_speech_actors_01-24/Actor_14", "audio_speech_actors_01-24/Actor_15", "audio_speech_actors_01-24/Actor_16", "audio_speech_actors_01-24/Actor_17", "audio_speech_actors_01-24/Actor_18", "audio_speech_actors_01-24/Actor_19", "audio_speech_actors_01-24/Actor_20", "audio_speech_actors_01-24/Actor_21", "audio_speech_actors_01-24/Actor_22", "audio_speech_actors_01-24/Actor_23", "audio_speech_actors_01-24/Actor_24"]

features = []
labels = []

for data_dir in dataset_dirs:
    for file in os.listdir(data_dir):
        if file.endswith(".wav"):
            try:
                file_path = os.path.join(data_dir, file)
                mfccs = extract_features(file_path)
                emotion_label = int(file.split("-")[2])  # Adjust this if needed
                features.append(mfccs)
                labels.append(emotion_label)
            except Exception as e:
                print(f" Skipped file {file}: {e}")

# Training logic
print(f" Total samples processed: {len(features)}")

if len(features) > 0:
    X = np.array(features)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    print(" Model trained. Accuracy:", model.score(X_test, y_test))
    joblib.dump(model, "emotion_model.pkl")
    print(" Model saved.")
else:
    print(" No data loaded. Check your folders.")
