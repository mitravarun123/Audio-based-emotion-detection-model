from flask import Flask, request, render_template, url_for
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = 'static/plots'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

model = load_model('saved_model.keras')
emotions = ['Angry' 'Disgusted' 'Fearful' 'Happy' 'Neutral' 'Sad' 'Suprised']  

def extract_features(file_path, n_mfcc=40, target_length=120, noise_factor=0.005):
    y, sr = librosa.load(file_path, sr=16000)
    noise = np.random.randn(len(y))
    y_noisy = y + noise_factor * noise
    mfcc = librosa.feature.mfcc(y=y_noisy, sr=sr, n_mfcc=n_mfcc)
    scaler = StandardScaler()
    mfcc = scaler.fit_transform(mfcc.T).T
    if mfcc.shape[1] < target_length:
        padding = np.zeros((n_mfcc, target_length - mfcc.shape[1]))
        mfcc = np.hstack((mfcc, padding))
    elif mfcc.shape[1] > target_length:
        mfcc = mfcc[:, :target_length]
    mfcc = np.expand_dims(mfcc, axis=-1)
    mfcc = np.expand_dims(mfcc, axis=0)
    return mfcc

def plot_audio_features(file_path,filename):
    y, sr = librosa.load(file_path)

    plt.figure(figsize=(14, 12))

    # 1. Waveform
    plt.subplot(3, 2, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')

    # 2. Spectrogram (log power)
    plt.subplot(3, 2, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format="%+2.0f dB")
    plt.title('Spectrogram')

    # 3. MFCCs
    plt.subplot(3, 2, 3)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title('MFCCs')

    # 4. Chroma
    plt.subplot(3, 2, 4)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr)
    plt.colorbar()
    plt.title('Chroma Feature')

    # 5. Spectral Centroid
    plt.subplot(3, 2, 5)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    frames = range(len(spec_centroid))
    t = librosa.frames_to_time(frames)
    librosa.display.waveshow(y, sr=sr, alpha=0.4)
    plt.plot(t, spec_centroid, color='r')
    plt.title('Spectral Centroid')

    # 6. Pitch (F0)
    plt.subplot(3, 2, 6)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                                 fmax=librosa.note_to_hz('C7'))
    times = librosa.times_like(f0)
    plt.plot(times, f0, label='f0', color='g')
    plt.title('Estimated Pitch (F0)')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    plt.tight_layout()
    plt.show()
    plot_path = os.path.join(PLOT_FOLDER, f"{filename}.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected")

    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    plot_path = plot_audio_features(filepath, os.path.splitext(filename)[0])
    plot_url = url_for('static', filename=f"plots/{os.path.splitext(filename)[0]}.png")

    features = extract_features(filepath)
    prediction = model.predict(features)
    pred_idx = np.argmax(prediction, axis=1)[0]
    predicted_emotion = emotions[pred_idx]

    return render_template('index.html', prediction=predicted_emotion, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
