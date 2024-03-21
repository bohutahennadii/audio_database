import os
import librosa
import matplotlib.pyplot as plt

def analyze_audio_lengths(folder):
    lengths = []
    for filename in os.listdir(folder):
        if filename.endswith('.wav'):
            filepath = os.path.join(folder, filename)
            y, sr = librosa.load(filepath, sr=None)
            lengths.append(librosa.get_duration(y=y, sr=sr))
    return lengths

generated_audio_lengths = analyze_audio_lengths('generated/')
print("Generated complete")
real_audio_lengths = analyze_audio_lengths('real/')
print("Real complete")

plt.figure(figsize=(6, 3))
plt.rcParams['font.size'] = 35  
plt.hist(generated_audio_lengths, bins=50, alpha=0.5, label='Generated Audio')
plt.hist(real_audio_lengths, bins=50, alpha=0.5, label='Real Audio')
plt.xlim(right = 60)
plt.xlabel('Length (seconds)')
plt.ylabel('Count')
plt.title('Distribution of Audio Lengths')
plt.legend()
plt.show()

plt.figure(figsize=(6, 3))
plt.rcParams['font.size'] = 35  
plt.boxplot([generated_audio_lengths, real_audio_lengths], labels=['Generated Audio', 'Real Audio'])
plt.ylabel('Length (seconds)')
plt.title('Box Plot of Audio Lengths')
plt.show()
