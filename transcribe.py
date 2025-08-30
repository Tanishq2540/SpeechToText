from faster_whisper import WhisperModel

def main():
    model = WhisperModel("medium", device="cpu")

    # Enable translation
    segments, info = model.transcribe("audio/amplifiedHU.wav", task="translate")

    print("Detected language:", info.language)
    print("Language probability:", info.language_probability)

    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

if __name__ == "__main__":
    main()
