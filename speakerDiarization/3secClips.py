import librosa
import numpy as np
import torch
from sklearn.cluster import KMeans
from speechbrain.pretrained import EncoderClassifier
import warnings
import os

# 1. Ignore all Python warnings
warnings.filterwarnings("ignore")

# 2. Silence HuggingFace / Transformers / Pyannote warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # TensorFlow-related (if used)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYANNOTE_AUDIO_PROGRESS_BAR"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 3. Optional: Disable torch & torchaudio info logs
import logging
logging.getLogger("pyannote.audio").setLevel(logging.ERROR)
logging.getLogger("torchaudio").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("speechbrain").setLevel(logging.ERROR)


# Load speaker embedding model
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# Load audio file
audio_file = "audio/aksAmplified.wav"
signal, sr = librosa.load(audio_file, sr=16000)  # resample to 16kHz

# --- STEP 1: Segment into chunks ---
chunk_size = 3 * sr  # 3 seconds
chunks = [signal[i:i+chunk_size] for i in range(0, len(signal), chunk_size)]

# --- STEP 2: Extract embeddings ---


embeddings = []
for chunk in chunks:
    if len(chunk) < sr:
        continue
    # convert numpy → tensor with batch dimension
    tensor_chunk = torch.tensor(chunk).unsqueeze(0)  # shape: (1, time)
    emb = classifier.encode_batch(tensor_chunk)
    emb = emb.squeeze().detach().cpu().numpy()       # shape: (D,)
    embeddings.append(emb)

embeddings = np.vstack(embeddings)  # final shape: (num_chunks, D)

# --- STEP 3: Cluster speakers ---
kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings)
labels = kmeans.labels_

# --- STEP 4: Print results ---
for i, label in enumerate(labels):
    start = i * 3
    end = start + 3
    print(f"{start:>3}s - {end:>3}s : Speaker {label}")
