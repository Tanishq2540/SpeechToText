from pyannote.audio import Pipeline
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


# works with pyannote.audio >= 3.x
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token=True)  # <-- your HF token already cached

# Run on your audio file
audio_file = "audio/sample.wav"   # can also be wav

vad_result = pipeline(audio_file)

# Print speech segments
for segment, _, label in vad_result.itertracks(yield_label=True):
    print(f"Speech: {segment.start:.1f}s --> {segment.end:.1f}s")
