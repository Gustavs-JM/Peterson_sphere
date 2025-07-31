from peterson_sphere_function import *
from whisperx_transcription import transcribe_basic
import os
from pathlib import Path

keys = open('keys.txt')
key_list = keys.readlines()
HF_TOKEN = key_list[1]

""" 
result = transcribe_with_diarization(
    audio_file="database/Jordan B Peterson/2013_03_30_A5216ZJVbVs_WhatMatters/audio_2013_03_30_A5216ZJVbVs_WhatMatters.wav",
    whisper_model="large-v3",
    language="en"
)

print(result)
if result["success"]:
    print(result["transcription"])
"""

# Call the function
## Change the
def main(HF_TOKEN):
    audio_file = "database/Jordan B Peterson/2025_05_09_QPp6fD_zzZM_ParentingTheOfficial/audio_2025_05_09_QPp6fD_zzZM_ParentingTheOfficial.mp3"  # Replace with your audio file path
    if not os.path.exists(audio_file):
        print(f"File not found: {audio_file}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")
    try:
        result = transcribe_basic(
            audio_file=audio_file,
            model_size="large-v2",  # or "medium", "small", etc.
            device="cpu",  # or "cpu" if no GPU, original "cuda"
            compute_type="int8", # This exists because I was running on my small device
        )
        if result["success"]:
            print(f"Transcription completed!")
            print(f"Language detected: {result['language']}")
            print("\nTranscript with speakers:")
            print("-" * 50)

            for segment in result["segments"]:
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                text = segment["text"]
                speaker = segment.get("speaker", "Unknown")

                print(f"[{start_time:.2f}s - {end_time:.2f}s] {speaker}: {text}")

        else:
            print("Transcription failed!")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main(HF_TOKEN)