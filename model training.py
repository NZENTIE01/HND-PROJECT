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
dataset_dirs = ["audiofile/Actor_01", "audiofile/Actor_02", "audiofile/Actor_03", "audiofile/Actor_04", "audiofile/Actor_05", "audiofile/Actor_06", "audiofile/Actor_07", "audiofile/Actor_08", "audiofile/Actor_09", "audiofile/Actor_10", "audiofile/Actor_11", "audiofile/Actor_12", "audiofile/Actor_13", "audiofile/Actor_14", "audiofile/Actor_15", "audiofile/Actor_16", "audiofile/Actor_17", "audiofile/Actor_18", "audiofile/Actor_19", "audiofile/Actor_20", "audiofile/Actor_21", "audiofile/Actor_22", "audiofile/Actor_23", "audiofile/Actor_24", "audiofile/audio_speech_actors_01-24/Actor_01", "audiofile/audio_speech_actors_01-24/Actor_02", "audiofile/audio_speech_actors_01-24/Actor_03", "audiofile/audio_speech_actors_01-24/Actor_04", "audiofile/audio_speech_actors_01-24/Actor_05", "audiofile/audio_speech_actors_01-24/Actor_06", "audiofile/audio_speech_actors_01-24/Actor_07", "audiofile/audio_speech_actors_01-24/Actor_08", "audiofile/audio_speech_actors_01-24/Actor_09", "audiofile/audio_speech_actors_01-24/Actor_10", "audiofile/audio_speech_actors_01-24/Actor_11", "audiofile/audio_speech_actors_01-24/Actor_12", "audiofile/audio_speech_actors_01-24/Actor_13", "audiofile/audio_speech_actors_01-24/Actor_14", "audiofile/audio_speech_actors_01-24/Actor_15", "audiofile/audio_speech_actors_01-24/Actor_16", "audiofile/audio_speech_actors_01-24/Actor_17", "audiofile/audio_speech_actors_01-24/Actor_18", "audiofile/audio_speech_actors_01-24/Actor_19", "audiofile/audio_speech_actors_01-24/Actor_20", "audiofile/audio_speech_actors_01-24/Actor_21", "audiofile/audio_speech_actors_01-24/Actor_22", "audiofile/audio_speech_actors_01-24/Actor_23", "audiofile/audio_speech_actors_01-24/Actor_24"]

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
