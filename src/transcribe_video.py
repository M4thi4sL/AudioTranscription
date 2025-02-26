import os
import whisper
import subprocess
from dotenv import load_dotenv
from tqdm import tqdm


def seconds_to_srt_time(seconds):
    """Convert seconds to SRT time format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


# Load environment variables from .env file
load_dotenv()

MODEL = whisper.load_model("small.en")


def extract_audio(video_file, output_audio_file):
    """Extracts audio from a video file using ffmpeg."""
    command = ["ffmpeg", "-i", video_file, "-ac", "1", "-ar", "16000", "-vn", output_audio_file, "-y"]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def transcribe_audio(audio_file):
    """Transcribes the extracted audio file."""
    result = MODEL.transcribe(audio_file, fp16=False, word_timestamps=True)
    return result["segments"]


def save_srt(segments, output_srt_file):
    """Saves transcribed segments in SRT format."""
    with open(output_srt_file, "w", encoding="utf-8") as srt_file:
        for idx, segment in enumerate(segments, start=1):
            start_time = seconds_to_srt_time(segment["start"])
            end_time = seconds_to_srt_time(segment["end"])
            text = segment["text"]

            srt_file.write(f"{idx}\n{start_time} --> {end_time}\n{text}\n\n")


def main():
    folder_path = os.getenv("FOLDER_PATH")
    extension = os.getenv("EXTENSION")  # Extension for video files (e.g., mp4, mkv, etc.)

    video_files = [file for file in os.listdir(folder_path) if file.lower().endswith(extension)]

    with tqdm(total=len(video_files), desc="Processing Videos") as pbar:
        for file in video_files:
            video_path = os.path.join(folder_path, file)
            audio_path = os.path.join(folder_path, "temp_audio.wav")
            output_srt_file = os.path.join(folder_path, os.path.splitext(file)[0] + ".srt")

            try:
                extract_audio(video_path, audio_path)
                segments = transcribe_audio(audio_path)
                save_srt(segments, output_srt_file)
            except Exception as e:
                print(f"Error processing {file}: {e}")
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)  # Clean up temporary audio file

            pbar.update(1)

    print("Subtitle generation completed.")


if __name__ == "__main__":
    main()
