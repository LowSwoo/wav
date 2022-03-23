import librosa
import matplotlib.pyplot as plt
import librosa.display
import sklearn
audio_data = './test.wav'
x, sr = librosa.load(audio_data)
librosa.beat
# print(x.shape, sr)
# plt.figure(figsize=(14, 5))
# librosa.display.waveshow(x, sr=sr)
# X = librosa.stft(x)
# Xdb = librosa.amplitude_to_db(abs(X))
# plt.figure(figsize=(14, 5))
# librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
# plt.colorbar()
# import numpy as np
# sr = 22050 # частота дискретизации
# T = 15.0    # секунды
# t = np.linspace(0, T, int(T*sr), endpoint=False) # переменная времени
# x = 0.5*np.sin(2*np.pi*220*t)

spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
spectral_centroids.shape
# (775,)
# # Вычисление временной переменной для визуализации
# plt.figure(figsize=(12, 4))
#
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
# # Нормализация спектрального центроида для визуализации
def normalize(x, axis=0):
     return sklearn.preprocessing.minmax_scale(x, axis=axis)
# # Построение спектрального центроида вместе с формой волны
# librosa.display.waveshow(x, sr=sr, alpha=0.4)
# plt.plot(t, normalize(spectral_centroids), color='b')
spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr)[0]


plt.figure(figsize=(15, 9))

librosa.display.waveshow(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_bandwidth_2), color='r')
# plt.plot(t, normalize(spectral_bandwidth_2 * x), color='g')


print(len(x))
print(len(spectral_bandwidth_2))
print(sr)
# print(spectral_bandwidth_2)
# print(x)
# print(spectral_bandwidth_2)

plt.legend(('p = 2', 'p = 3', 'p = 4'))
plt.show()