import os
import numpy as np
import soundfile as sf
import librosa
import scipy.signal as signal
from scipy.signal import wiener, butter, lfilter

# ===================== CONFIG =====================
input_file = "schrodinger.wav"
output_dir = "outputs"
spectral_dir = "outputs_spectral"   # amplified spectral will be saved here
os.makedirs(output_dir, exist_ok=True)
os.makedirs(spectral_dir, exist_ok=True)

# ===================== LOAD AUDIO =====================
y, sr = librosa.load(input_file, sr=None)

# ===================== WIENER FILTER =====================
y_wiener = wiener(y)
sf.write(os.path.join(output_dir, "wiener.wav"), y_wiener, sr)

# ===================== SPECTRAL SUBTRACTION =====================
D = librosa.stft(y)
magnitude, phase = librosa.magphase(D)

# Estimate noise spectrum (first 30 frames ≈ 0.5 sec if sr=16k)
noise_mag = np.mean(np.abs(magnitude[:, :30]), axis=1, keepdims=True)

# Subtract noise
clean_mag = np.maximum(np.abs(magnitude) - noise_mag, 0.0)
clean_D = clean_mag * phase
y_spectral = librosa.istft(clean_D)

# === Amplify spectral ===
y_spectral_amplified = y_spectral * 2.0   # amplify factor (can adjust 1.5–3.0)
sf.write(os.path.join(spectral_dir, "spectral_amplified_schrodinger.wav"), y_spectral_amplified, sr)

# ===================== BAND-PASS FILTER =====================
lowcut, highcut = 300, 3400  # speech range
b, a = butter(6, [lowcut/(sr/2), highcut/(sr/2)], btype='band')
y_bandpass = lfilter(b, a, y)
sf.write(os.path.join(output_dir, "bandpass.wav"), y_bandpass, sr)

print("✅ All files saved:")
print(f"- Wiener: {os.path.join(output_dir, 'wiener.wav')}")
print(f"- Band-pass: {os.path.join(output_dir, 'bandpass.wav')}")
print(f"- Spectral (amplified): {os.path.join(spectral_dir, 'spectral_amplified.wav')}")
