import scipy.stats as stats
import os
import librosa

def compute_statistics(data):
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    minimum = np.min(data)
    maximum = np.max(data)
    return mean, median, std_dev, minimum, maximum

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

generated_stats = compute_statistics(generated_audio_lengths)
real_stats = compute_statistics(real_audio_lengths)

print("Generated Audio Statistics:")
print("Mean:", generated_stats[0])
print("Median:", generated_stats[1])
print("Standard Deviation:", generated_stats[2])
print("Minimum:", generated_stats[3])
print("Maximum:", generated_stats[4])

print("\nReal Audio Statistics:")
print("Mean:", real_stats[0])
print("Median:", real_stats[1])
print("Standard Deviation:", real_stats[2])
print("Minimum:", real_stats[3])
print("Maximum:", real_stats[4])

t_stat, p_value = stats.ttest_ind(generated_audio_lengths, real_audio_lengths, equal_var=False)

print("\nT-test Results:")
print("T-statistic:", t_stat)
print("P-value:", p_value)
