from pyannote.audio import Pipeline
from pyannote.core import Segment
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

# 🔑 Make sure you have logged in:
# huggingface-cli login

# Load diarization pipeline (3.x)
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", use_auth_token=True
)

# Run diarization on your audio file
diarization = diarization_pipeline("audio/huClassSample.wav")

print("\n--- Speaker Diarization ---\n")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    # turn is a Segment object in 3.x
    print(f"{turn.start:.1f}s - {turn.end:.1f}s: {speaker}")

# Load VAD pipeline (3.x)
vad_pipeline = Pipeline.from_pretrained(
    "pyannote/voice-activity-detection", use_auth_token=True
)

vad = vad_pipeline("audio/huClassSample.wav")

print("\n--- Voice Activity Detection (speech segments) ---\n")
for speech in vad.get_timeline().support():
    # speech is a Segment object
    print(f"{speech.start:.1f}s - {speech.end:.1f}s: Speech")
