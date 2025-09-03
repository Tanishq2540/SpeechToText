import os
import torch
import json
from faster_whisper import WhisperModel
from speechbrain.inference import SpeakerRecognition   # updated import
from pydub import AudioSegment

# ---------------- CONFIG ----------------
AUDIO_FILE = "audio/math_demo.wav"   # Full noisy recording
PROFESSOR_REF = "audio/clean_tanmay.wav"   # Short clean sample
WHISPER_MODEL = "medium"                             # Try "large-v2" for Hinglish

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
print(f"Using device: {DEVICE} | compute_type: {COMPUTE_TYPE}")

# Load models
whisper_model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
verifier = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE}
)

# ---------------- FUNCTIONS ----------------
def save_segment(input_audio, output_file, start, end):
    """Save an audio segment using pydub instead of ffmpeg binary"""
    audio = AudioSegment.from_file(input_audio)
    segment = audio[start * 1000:end * 1000]  # pydub works in milliseconds
    segment = segment.set_channels(1).set_frame_rate(16000)  # mono, 16kHz
    segment.export(output_file, format="wav")

def is_professor(segment_file, threshold=0.35):
    """Check if a segment belongs to professor using speaker verification"""
    score, prediction = verifier.verify_files(PROFESSOR_REF, segment_file)
    return prediction and score > threshold

def transcribe_and_filter(audio_path):
    """Transcribe audio and keep only professor's speech"""
    print("1) Transcribing audio with Faster-Whisper...")
    segments, info = whisper_model.transcribe(audio_path, task="translate")
    print(f"   -> Detected language: {info.language} (p={info.language_probability:.2f})")

    results = []
    for seg in segments:
        tmp_file = f"chunk_{int(seg.start*100)}.wav"
        save_segment(audio_path, tmp_file, seg.start, seg.end)

        if is_professor(tmp_file):
            results.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip()
            })

        os.remove(tmp_file)  # cleanup

    return results

def save_transcript(transcript, json_file="prof_transcript.json", txt_file="prof_transcript.txt"):
    """Save transcript in JSON + TXT formats"""
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2)

    with open(txt_file, "w", encoding="utf-8") as f:
        for seg in transcript:
            f.write(f"[{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['text']}\n")

    print(f"\n📂 Saved transcript to {json_file} and {txt_file}")

# ---------------- MAIN ----------------
def main():
    transcript = transcribe_and_filter(AUDIO_FILE)

    print("\n=== PROFESSOR-ONLY TRANSCRIPT ===\n")
    for seg in transcript:
        print(f"[{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['text']}")

    save_transcript(transcript)

if __name__ == "__main__":
    main()
