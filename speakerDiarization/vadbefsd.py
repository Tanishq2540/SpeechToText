import os
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# --------------- CONFIG ----------------
AUDIO_FILE = "audio/amplifiedHU.wav"
DEVICE = "cpu"
WHISPER_MODEL = "medium"  # latest Faster-Whisper model
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
HF_TOKEN = "HF_TOKEN"  # set your HF token here or via huggingface-cli login

# ----------------------------------------

def transcribe_audio(audio_path):
    """Transcribe audio with Faster-Whisper"""
    print("1) Transcribing audio with Faster-Whisper...")
    model = WhisperModel(WHISPER_MODEL, device=DEVICE)
    segments, info = model.transcribe(audio_path, task="translate")  # use task="transcribe" if you want original language
    transcript = []
    for seg in segments:
        transcript.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip()
        })
    print(f"   -> Detected language: {info.language} (p={info.language_probability:.2f})")
    return transcript

def diarize_audio(audio_path):
    """Perform speaker diarization with Pyannote"""
    print("2) Running speaker diarization with Pyannote...")
    pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=HF_TOKEN)
    diarization = pipeline(audio_path)
    speakers = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    return speakers

def merge_transcript_and_speakers(transcript, speakers):
    """
    Ultra-optimized merging of transcription + diarization outputs.
    Uses two-pointer traversal with early exit for O(N + M) complexity.
    """
    merged = []
    i, j = 0, 0
    n, m = len(transcript), len(speakers)

    while i < n:
        seg = transcript[i]
        start, end = seg["start"], seg["end"]
        matched_speaker = "Unknown"
        max_overlap = 0

        # Move the speaker pointer forward if current speaker ends before transcript starts
        while j < m and speakers[j]["end"] <= start:
            j += 1

        # Check overlap with relevant speakers only
        k = j
        while k < m:
            spk = speakers[k]

            # If this speaker starts after the transcript ends, break early
            if spk["start"] >= end:
                break

            # Calculate overlap duration
            overlap = min(end, spk["end"]) - max(start, spk["start"])
            if overlap > max_overlap:
                max_overlap = overlap
                matched_speaker = spk["speaker"]

            k += 1

        # Store the merged result
        merged.append({
            "start": start,
            "end": end,
            "speaker": matched_speaker,
            "text": seg["text"]
        })

        i += 1

    return merged


def main():
    transcript = transcribe_audio(AUDIO_FILE)
    speakers = diarize_audio(AUDIO_FILE)
    merged_output = merge_transcript_and_speakers(transcript, speakers)

    print("\n=== FINAL SPEAKER-LABELLED TRANSCRIPT ===\n")
    for seg in merged_output:
        print(f"[{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['speaker']}: {seg['text']}")

if __name__ == "__main__":
    import torch
    main()
