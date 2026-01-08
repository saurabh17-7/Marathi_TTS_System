import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
audio_path = os.path.join(BASE_DIR, "output.wav")

print("Using audio file:", audio_path)

# ---------------- LOAD AUDIO ----------------
audio, sr = sf.read(audio_path)

# Convert stereo to mono if needed
if len(audio.shape) > 1:
    audio = audio.mean(axis=1)

# ---------------- TIME DOMAIN WAVEFORM ----------------
time = np.linspace(0, len(audio) / sr, len(audio))

plt.figure(figsize=(10, 4))
plt.plot(time, audio)
plt.title("Speech Signal Waveform (Time Domain)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()
plt.tight_layout()
plt.show()

# ---------------- FREQUENCY DOMAIN (FFT) ----------------
N = len(audio)
yf = fft(audio)
xf = fftfreq(N, 1 / sr)

plt.figure(figsize=(10, 4))
plt.plot(xf[:N // 2], np.abs(yf[:N // 2]))
plt.title("Frequency Spectrum of Speech Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.tight_layout()
plt.show()

# ---------------- ENERGY CALCULATION ----------------
energy = np.sum(audio ** 2)
print("Signal Energy:", energy)

# ---------------- SPECTROGRAM ----------------
plt.figure(figsize=(10, 4))
plt.specgram(audio, Fs=sr, NFFT=1024, noverlap=512, cmap="jet")
plt.title("Spectrogram of Marathi Speech Signal")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(label="Intensity (dB)")
plt.tight_layout()
plt.show()
