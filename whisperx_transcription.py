import whisperx
import gc


def transcribe_basic(
        audio_file: str,
        model_size: str = "large-v2",
        device: str = "cpu",
        compute_type: str = "int8"
):
    """
    Most basic transcription - just Whisper without alignment complications.
    """

    print(f"Loading model: {model_size}")
    model = whisperx.load_model(model_size, device, compute_type=compute_type)

    print(f"Loading audio: {audio_file}")
    audio = whisperx.load_audio(audio_file)

    print("Starting transcription...")
    result = model.transcribe(audio, batch_size=16)

    print(f"Transcription complete. Language detected: {result['language']}")

    # Skip alignment to avoid issues
    print("Skipping alignment - using Whisper's original timestamps")

    # Cleanup
    del model
    gc.collect()

    return {
        "success": True,
        "segments": result["segments"],
        "language": result["language"],
        "full_text": " ".join([seg["text"] for seg in result["segments"]])
    }


# Example usage
if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = input("Enter path to audio file: ")

    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"Error: File not found: {audio_file}")
        exit(1)

    print(f"Processing: {audio_file}")

    try:
        result = transcribe_basic(audio_file)

        if result["success"]:
            print(f"\n✅ SUCCESS!")
            print(f"Language: {result['language']}")
            print(f"Total segments: {len(result['segments'])}")

            # Save transcript to file
            output_file = audio_file.rsplit('.', 1)[0] + '_transcript.txt'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Transcript - Language: {result['language']}\n")
                f.write("=" * 50 + "\n\n")
                f.write(result['full_text'] + "\n\n")
                f.write("Segments with timestamps:\n")
                f.write("-" * 30 + "\n")
                for segment in result['segments']:
                    start = segment.get('start', 0)
                    end = segment.get('end', 0)
                    text = segment['text']
                    f.write(f"[{start:.2f}s - {end:.2f}s]: {text}\n")

            print(f"\n📄 Transcript saved to: {output_file}")

            # Print first few segments
            print(f"\n📝 Preview (first 3 segments):")
            for i, segment in enumerate(result['segments'][:3]):
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                text = segment['text']
                print(f"  [{start:.2f}s - {end:.2f}s]: {text}")

            if len(result['segments']) > 3:
                print(f"  ... and {len(result['segments']) - 3} more segments")

    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        import traceback

        traceback.print_exc()