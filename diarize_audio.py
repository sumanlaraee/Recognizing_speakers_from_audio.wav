import os
import sys
import argparse
import subprocess
import tempfile
from pyannote.audio import Pipeline

def convert_to_wav(input_path: str) -> str:
    """
    Convert any audio file to mono 16 kHz WAV using ffmpeg.
    Returns the path to the converted WAV file.
    """
    # create a temporary WAV file
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav_path = tmp_wav.name
    tmp_wav.close()

    cmd = [
        "ffmpeg",
        "-y",                   # overwrite if exists
        "-i", input_path,       # input file
        "-ac", "1",             # mono
        "-ar", "16000",         # 16 kHz
        "-vn",                  # no video
        tmp_wav_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        os.unlink(tmp_wav_path)
        raise RuntimeError(f"ffmpeg failed to convert {input_path} to WAV")

    return tmp_wav_path

def diarize_audio(file_path: str, hf_token: str):
    """
    Perform speaker diarization on `file_path` (any format) and print:
      - Number of unique speakers
      - Time-stamped speaker turns
    """
    # 1. Ensure WAV input
    ext = os.path.splitext(file_path)[1].lower()
    if ext != ".wav":
        wav_path = convert_to_wav(file_path)
    else:
        wav_path = file_path

    # 2. Load pretrained diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )

    # 3. Apply diarization
    diarization = pipeline(wav_path)

    # 4. Extract segments and speaker labels
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 2),
            "end":   round(turn.end,   2)
        })

    # 5. Count distinct speakers
    speakers = sorted({seg["speaker"] for seg in segments})

    # 6. Output results
    print(f"\nDetected {len(speakers)} speakers: {', '.join(speakers)}\n")
    print("Speaker turns:")
    for seg in segments:
        print(f"  {seg['speaker']}: {seg['start']}s – {seg['end']}s")

    # 7. Clean up temporary file if created
    if ext != ".wav":
        os.unlink(wav_path)

    return segments

def main():
    parser = argparse.ArgumentParser(
        description="Unsupervised speaker diarization using pyannote.audio"
    )
    parser.add_argument(
        "audio_file",
        help="Path to the input audio file (e.g., your_clip.m4a or .wav)"
    )
    parser.add_argument(
        "--hf_token",
        help="Hugging Face token (default: read from HF_TOKEN env var)",
        default=os.getenv("HF_TOKEN")
    )
    args = parser.parse_args()

    # Validate audio file
    if not os.path.isfile(args.audio_file):
        print(f"Error: audio file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    # Validate token
    if not args.hf_token:
        print(
            "Error: Hugging Face token not provided.\n"
            "  • Set the HF_TOKEN environment variable, or\n"
            "  • pass --hf_token your_token_here",
            file=sys.stderr
        )
        sys.exit(1)

    try:
        diarize_audio(args.audio_file, args.hf_token)
    except Exception as e:
        print(f"Error during diarization: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
