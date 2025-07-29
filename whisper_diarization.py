import os
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, List, Union


class WhisperDiarization:
    """
    A class to handle whisper-diarization transcription with speaker identification.

    Requires the whisper-diarization repository to be cloned and dependencies installed.
    """

    def __init__(self, whisper_diarization_path: str = "./whisper-diarization"):
        """
        Initialize the WhisperDiarization class.

        Args:
            whisper_diarization_path: Path to the cloned whisper-diarization repository
        """
        self.repo_path = Path(whisper_diarization_path)
        self.diarize_script = self.repo_path / "diarize.py"

        if not self.diarize_script.exists():
            raise FileNotFoundError(f"diarize.py not found at {self.diarize_script}")

    def transcribe(
            self,
            audio_file: str,
            whisper_model: str = "medium.en",
            language: Optional[str] = None,
            device: str = "cuda",
            no_stem: bool = False,
            suppress_numerals: bool = False,
            batch_size: int = 8,
            output_dir: Optional[str] = None
    ) -> Dict[str, Union[str, List]]:
        """
        Transcribe audio file with speaker diarization.

        Args:
            audio_file: Path to the audio file to transcribe
            whisper_model: Whisper model to use (default: "medium.en")
            language: Language code (e.g., "en", "es", "fr"). Auto-detect if None
            device: Device to use ("cuda", "cpu", or specific GPU like "cuda:0")
            no_stem: If True, disables source separation
            suppress_numerals: If True, transcribes numbers as words instead of digits
            batch_size: Batch size for inference (reduce if out of memory, 0 for non-batched)
            output_dir: Directory to save output files (uses audio file directory if None)

        Returns:
            Dictionary containing transcription results and file paths
        """

        # Validate input file
        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        # Set output directory
        if output_dir is None:
            output_dir = audio_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            "python", str("diarize.py"),
            "-a", str(audio_path),
            "--whisper-model", whisper_model,
            "--device", device,
            "--batch-size", str(batch_size)
        ]

        print(str(self.diarize_script))

        if no_stem:
            cmd.append("--no-stem")

        if suppress_numerals:
            cmd.append("--suppress_numerals")

        if language:
            cmd.extend(["--language", language])

        try:
            # Run the diarization
            print(f"Starting transcription of {audio_file}...")
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            print("Transcription completed successfully!")

            # Get output files
            audio_stem = audio_path.stem
            txt_file = output_dir / f"{audio_stem}.txt"
            srt_file = output_dir / f"{audio_stem}.srt"

            # Read transcription results
            transcription_text = ""
            if txt_file.exists():
                with open(txt_file, 'r', encoding='utf-8') as f:
                    transcription_text = f.read()

            # Parse SRT for structured data
            segments = []
            if srt_file.exists():
                segments = self._parse_srt(srt_file)

            return {
                "success": True,
                "transcription": transcription_text,
                "segments": segments,
                "txt_file": str(txt_file),
                "srt_file": str(srt_file),
                "stdout": result.stdout,
                "stderr": result.stderr
            }

        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"Transcription failed with exit code {e.returncode}",
                "stdout": e.stdout,
                "stderr": e.stderr
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def transcribe_parallel(
            self,
            audio_file: str,
            **kwargs
    ) -> Dict[str, Union[str, List]]:
        """
        Use parallel processing for systems with >=10GB VRAM.
        This runs NeMo in parallel with Whisper for better performance.

        Args:
            audio_file: Path to the audio file to transcribe
            **kwargs: Same arguments as transcribe() method

        Returns:
            Dictionary containing transcription results and file paths
        """
        parallel_script = self.repo_path / "diarize_parallel.py"
        if not parallel_script.exists():
            raise FileNotFoundError(f"diarize_parallel.py not found at {parallel_script}")

        # Use parallel script instead
        original_script = self.diarize_script
        self.diarize_script = parallel_script

        try:
            result = self.transcribe(audio_file, **kwargs)
        finally:
            # Restore original script
            self.diarize_script = original_script

        return result

    def _parse_srt(self, srt_file: Path) -> List[Dict]:
        """Parse SRT file to extract segments with timestamps and speakers."""
        segments = []

        try:
            with open(srt_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split by double newlines to get segments
            srt_segments = content.strip().split('\n\n')

            for segment in srt_segments:
                lines = segment.strip().split('\n')
                if len(lines) >= 3:
                    # Parse segment number
                    seg_num = lines[0]

                    # Parse timestamps
                    timestamps = lines[1]
                    start_time, end_time = timestamps.split(' --> ')

                    # Parse text (may span multiple lines)
                    text = '\n'.join(lines[2:])

                    # Extract speaker if present (format: "SPEAKER_XX: text")
                    speaker = "Unknown"
                    if ': ' in text:
                        potential_speaker, remaining_text = text.split(': ', 1)
                        if potential_speaker.startswith('SPEAKER_'):
                            speaker = potential_speaker
                            text = remaining_text

                    segments.append({
                        "segment_id": seg_num,
                        "start_time": start_time,
                        "end_time": end_time,
                        "speaker": speaker,
                        "text": text.strip()
                    })

        except Exception as e:
            print(f"Warning: Could not parse SRT file: {e}")

        return segments


# Convenience function for quick transcription
def transcribe_with_diarization(
        audio_file: str,
        whisper_diarization_path: str = "./whisper-diarization",
        **kwargs
) -> Dict[str, Union[str, List]]:
    """
    Quick function to transcribe audio with speaker diarization.

    Args:
        audio_file: Path to audio file
        whisper_diarization_path: Path to whisper-diarization repository
        **kwargs: Additional arguments for transcription

    Returns:
        Dictionary with transcription results
    """
    transcriber = WhisperDiarization(whisper_diarization_path)
    return transcriber.transcribe(audio_file, **kwargs)


# Example usage
if __name__ == "__main__":
    # Initialize the transcriber
    transcriber = WhisperDiarization(".")

    # Transcribe an audio file
    result = transcriber.transcribe(
        audio_file="path/to/your/audio.wav",
        whisper_model="large-v3",
        language="en",
        device="cuda",
        no_stem=True
    )

    if result["success"]:
        print("Transcription successful!")
        print(f"Text file: {result['txt_file']}")
        print(f"SRT file: {result['srt_file']}")
        print("\nTranscription:")
        print(result["transcription"])

        print("\nSegments with speakers:")
        for segment in result["segments"]:
            print(f"{segment['speaker']} ({segment['start_time']} -> {segment['end_time']}): {segment['text']}")
    else:
        print(f"Transcription failed: {result['error']}")
        if "stderr" in result:
            print(f"Error details: {result['stderr']}")