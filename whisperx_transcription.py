import whisperx
import gc
import torch

import whisperx
import gc
import torch


def transcribe_basic(
    audio_file: str,
    model_size: str = "medium",
    device: str = "cpu",
    compute_type: str = "int8",
    batch_size: int = 8,
    hf_token: str = None,
    min_speakers: int = None,
    max_speakers: int = None,
    num_speakers: int = None
):
    """
    WhisperX transcription with diarization that works across all versions.
    Uses pyannote directly instead of relying on WhisperX's DiarizationPipeline.
    """

    print(f"🎤 Processing: {audio_file}")
    print(f"Settings: {model_size}, {device}, {compute_type}, batch_size={batch_size}")

    # Step 1: Load Whisper model and transcribe
    print("\n1️⃣ Loading Whisper model...")
    model = whisperx.load_model(model_size, device, compute_type=compute_type)

    print("2️⃣ Loading audio...")
    audio = whisperx.load_audio(audio_file)

    print("3️⃣ Transcribing audio...")
    result = model.transcribe(audio, batch_size=batch_size)

    # Check if language was detected
    if 'language' not in result:
        print("⚠️ Language detection failed, defaulting to English")
        result['language'] = 'en'

    print(f"✅ Transcription complete! Language: {result['language']}")

    # Clean up transcription model
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Step 2: Word-level alignment
    print("\n4️⃣ Aligning words...")
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False
        )
        print("✅ Word alignment complete!")

        # Clean up alignment model
        del model_a
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"⚠️ Alignment failed: {e}")
        print("Continuing with original timestamps...")

    # Step 3: Speaker Diarization using pyannote directly
    if hf_token:
        print("\n5️⃣ Starting speaker diarization...")
        try:
            # Import pyannote directly
            from pyannote.audio import Pipeline

            print("Loading diarization pipeline...")
            # Load the diarization pipeline directly from pyannote
            diarize_model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )

            # Move to device if using GPU
            if device != "cpu":
                diarize_model = diarize_model.to(torch.device(device))

            # Set up diarization parameters
            diarize_kwargs = {}
            if min_speakers is not None:
                diarize_kwargs['min_speakers'] = min_speakers
            if max_speakers is not None:
                diarize_kwargs['max_speakers'] = max_speakers
            if num_speakers is not None:
                diarize_kwargs['num_speakers'] = num_speakers

            print(f"Diarization parameters: {diarize_kwargs}")

            # Run diarization on the audio file directly
            print("Running diarization...")
            diarize_segments = diarize_model(audio_file, **diarize_kwargs)

            # Assign speaker labels to transcript segments
            print("Assigning speakers to transcript...")
            result = assign_speakers_to_segments(diarize_segments, result)

            # Count unique speakers found
            speakers = set()
            for segment in result["segments"]:
                speaker = segment.get("speaker")
                if speaker:
                    speakers.add(speaker)

            print(f"✅ Diarization complete! Found {len(speakers)} speakers: {list(speakers)}")

        except ImportError:
            print("❌ pyannote.audio not installed!")
            print("Install with: pip install pyannote.audio")

        except Exception as e:
            print(f"❌ Diarization failed: {e}")
            print("Possible issues:")
            print("1. Invalid HuggingFace token")
            print("2. Haven't accepted model agreements:")
            print("   - https://huggingface.co/pyannote/segmentation-3.0")
            print("   - https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("3. Network connectivity issues")
            print("4. Insufficient memory")
    else:
        print("\n⚠️ No HuggingFace token provided - skipping diarization")
        print("All speakers will be labeled as 'Unknown'")

    # Step 4: Prepare final results
    result['full_text'] = " ".join([seg["text"] for seg in result["segments"]])

    return {
        "success": True,
        "segments": result["segments"],
        "language": 'en',
        "full_text": result["full_text"],
        "audio_file": audio_file
    }

def save_diarized_transcript(result, output_file=None):
    """Save diarized transcript to text file"""
    if output_file is None:
        audio_file = result.get("audio_file", "audio")
        output_file = audio_file.rsplit('.', 1)[0] + '_diarized_transcript.txt'

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Diarized Transcript\n")
        f.write(f"Language: {result['language']}\n")
        f.write(f"Audio: {result.get('audio_file', 'Unknown')}\n")
        f.write("=" * 50 + "\n\n")

        # Write full text
        f.write("FULL TRANSCRIPT:\n")
        f.write(result['full_text'] + "\n\n")

        # Write segments with speakers
        f.write("SEGMENTS WITH SPEAKERS:\n")
        f.write("-" * 30 + "\n")

        for segment in result['segments']:
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            text = segment['text']
            speaker = segment.get('speaker', 'Unknown')

            f.write(f"[{start:.2f}s - {end:.2f}s] {speaker}: {text}\n")

    print(f"📄 Transcript saved to: {output_file}")
    return output_file


def create_srt_with_speakers(result, output_file=None):
    """Create SRT subtitle file with speaker labels"""
    if output_file is None:
        audio_file = result.get("audio_file", "audio")
        output_file = audio_file.rsplit('.', 1)[0] + '_diarized.srt'

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(result['segments'], 1):
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            text = segment['text']
            speaker = segment.get('speaker', 'Unknown')

            # Convert to SRT time format
            start_time = seconds_to_srt_time(start)
            end_time = seconds_to_srt_time(end)

            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{speaker}: {text}\n\n")

    print(f"🎬 SRT file saved to: {output_file}")
    return output_file


def seconds_to_srt_time(seconds):
    """Convert seconds to SRT time format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


def get_hf_token_instructions():
    """Print instructions for getting HuggingFace token"""
    print("\n🔑 HOW TO GET HUGGINGFACE TOKEN:")
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Click 'New token'")
    print("3. Choose 'Read' permissions")
    print("4. Copy the token (starts with 'hf_')")
    print("\n📋 ACCEPT MODEL AGREEMENTS:")
    print("5. Visit: https://huggingface.co/pyannote/segmentation-3.0")
    print("6. Click 'Agree and access repository'")
    print("7. Visit: https://huggingface.co/pyannote/speaker-diarization-3.1")
    print("8. Click 'Agree and access repository'")
    print("\nThen use your token in the function!")


def assign_speakers_to_segments(diarize_segments, transcript_result):
    """
    Assign speaker labels from diarization to transcript segments.
    This replaces whisperx.assign_word_speakers for compatibility.
    """

    # Convert diarization segments to a list of speaker intervals
    speaker_intervals = []
    for speech_turn, _, speaker in diarize_segments.itertracks(yield_label=True):
        speaker_intervals.append({
            'start': speech_turn.start,
            'end': speech_turn.end,
            'speaker': speaker
        })

    # Assign speakers to transcript segments based on overlap
    for segment in transcript_result["segments"]:
        segment_start = segment.get('start', 0)
        segment_end = segment.get('end', segment_start)
        segment_mid = (segment_start + segment_end) / 2

        # Find the speaker interval that contains the middle of this segment
        best_speaker = "Unknown"
        best_overlap = 0

        for speaker_interval in speaker_intervals:
            # Check if segment overlaps with speaker interval
            overlap_start = max(segment_start, speaker_interval['start'])
            overlap_end = min(segment_end, speaker_interval['end'])
            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration > best_overlap:
                best_overlap = overlap_duration
                best_speaker = speaker_interval['speaker']

        segment['speaker'] = best_speaker

        # Also assign speaker to individual words if they exist
        if 'words' in segment:
            for word in segment['words']:
                word['speaker'] = best_speaker

    return transcript_result