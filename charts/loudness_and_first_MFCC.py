import os
import librosa
import matplotlib.pyplot as plt
import numpy as np

def analyze_audio_features(folder):
    sample_rates = []
    loudness_values = []
    mfcc_values = []

    for filename in os.listdir(folder):
        if filename.endswith('.wav'):
            filepath = os.path.join(folder, filename)
            y, sr = librosa.load(filepath, sr=None)
            sample_rates.append(sr)

            loudness = np.mean(librosa.feature.rms(y=y))
            loudness_values.append(loudness)

            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            mfcc_values.append(np.mean(mfcc, axis=1))  

    return sample_rates, loudness_values, mfcc_values

generated_sample_rates, generated_loudness, generated_mfcc = analyze_audio_features('generated/')
print("Generated complete")
real_sample_rates, real_loudness, real_mfcc = analyze_audio_features('real/')
print("Real complete")

plt.figure(figsize=(12, 6))
plt.rcParams['font.size'] = 30  
plt.hist(generated_sample_rates, bins=np.unique(generated_sample_rates), alpha=0.5, label='Generated Audio')
plt.hist(real_sample_rates, bins=np.unique(real_sample_rates), alpha=0.5, label='Real Audio')
plt.xlabel('Sampling Rate (Hz)')
plt.ylabel('Count')
plt.title('Distribution of Sampling Rates')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.rcParams['font.size'] = 30
plt.hist(generated_loudness, bins=50, alpha=0.5, label='Generated Audio')
plt.hist(real_loudness, bins=50, alpha=0.5, label='Real Audio')
plt.xlabel('Loudness')
plt.ylabel('Count')
plt.title('Distribution of Loudness')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.rcParams['font.size'] = 30
plt.hist([mfcc[0] for mfcc in generated_mfcc], bins=50, alpha=0.5, label='Generated Audio')
plt.hist([mfcc[0] for mfcc in real_mfcc], bins=50, alpha=0.5, label='Real Audio')
plt.xlabel('First MFCC')
plt.ylabel('Count')
plt.title('Distribution of First MFCC')
plt.legend()
plt.show()
