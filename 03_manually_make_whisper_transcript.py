from peterson_sphere_function import *
from whisper_diarization import transcribe_with_diarization

result = transcribe_with_diarization(
    audio_file="database/Jordan B Peterson/2013_03_30_A5216ZJVbVs_WhatMatters/audio_2013_03_30_A5216ZJVbVs_WhatMatters.wav",
    whisper_model="large-v3",
    language="en"
)

print(result)
if result["success"]:
    print(result["transcription"])