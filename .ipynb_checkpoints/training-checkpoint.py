# -*- coding: utf-8 -*-
"""
Created on Fri May  2 18:44:20 2025

@author: MITRAVARUN
"""


import os
import pandas as pd  
import random
import matplotlib.pyplot as plt 
import seaborn as sns 
import librosa
import tensorflow as tf 
from keras.layers import Dense
import numpy as np
data_path = "D:/Emotions/Emotions"
filenames = os.listdir(data_path)
for name in filenames:
    print(f"Number of files in folder {name} are ",len(os.listdir((os.path.join(data_path,name)))))
    

#%%

def training_and_testing_data(data_path, folder_name, split_ratio):
    files = os.listdir(os.path.join(data_path, folder_name))
    total_files = len(files)
    value = int(split_ratio * total_files)

    train_files = files[:value]
    test_files = files[value:]

    train_data = pd.DataFrame({
        'Filepath': [os.path.join(data_path, folder_name, f) for f in train_files],
        'Emotion': [folder_name] * len(train_files)
    })

    test_data = pd.DataFrame({
        'Filepath': [os.path.join(data_path, folder_name, f) for f in test_files],
        'Emotion': [folder_name] * len(test_files)
    })

    return train_data, test_data


train_data = pd.DataFrame(columns=['Filepath', 'Emotion'])
test_data = pd.DataFrame(columns=['Filepath', 'Emotion'])


folder_names = os.listdir(data_path)
for folder_name in folder_names:
    curr_train, curr_test = training_and_testing_data(data_path, folder_name, 0.75)
    train_data = pd.concat([train_data, curr_train], ignore_index=True)
    test_data = pd.concat([test_data, curr_test], ignore_index=True)

train_data = train_data.sample(frac=1).reset_index(drop=True)
test_data = test_data.sample(frac=1).reset_index(drop=True)


#%%

sns.countplot(data=train_data,x='Emotion')
plt.title('Count of Emotions in trainging data')
plt.show()

sns.countplot(data=test_data,x='Emotion')
plt.title('Label count in testing data')
plt.show()
#%%
def plot_audio_features(file_path):
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
    return 
plot_audio_features(train_data['Filepath'][0])
#%%

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    # Basic features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_delta = librosa.feature.delta(mfccs)
    # mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    # contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    # tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    

    # tempo = librosa.beat.tempo(y=y, sr=sr)
    

    # f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
    #                                              fmax=librosa.note_to_hz('C7'))
    # f0 = f0[~np.isnan(f0)] 

    def summarize(x):
        return np.array([np.mean(x), np.std(x), np.min(x), np.max(x)])

    # Collect all summarized features
    feature_vector = np.hstack([
        summarize(mfccs),
        summarize(mfccs_delta),
        # summarize(mfccs_delta2),
        summarize(chroma),
        summarize(spectral_centroid),
        summarize(spectral_bandwidth),
        summarize(rolloff),
        summarize(zcr),
        summarize(rms),
        # summarize(contrast),
        # summarize(tonnetz),
        # summarize(f0) if len(f0) > 0 else np.zeros(4),  
        # tempo  
    ])

    return feature_vector
#%%
training_vectors = []
testing_vectors = []
train_paths = train_data['Filepath'].values 
test_paths = test_data['Filepath'].values 

for path in train_paths:
    vector = extract_features(path)
    training_vectors.append(vector)