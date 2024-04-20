import os
import whisper
import csv
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

MODEL = whisper.load_model("small.en")


def transcribe_wav_to_text(audio_file):
    result = MODEL.transcribe(audio_file, fp16=False)
    return result["text"], result["language"]


def main():
    folder_path = os.getenv("FOLDER_PATH")
    output_csv_file = os.getenv("OUTPUT_CSV_FILE")
    extension = os.getenv(key="EXTENSION")

    audio_files = [
        file for file in os.listdir(folder_path) if file.lower().endswith(extension)
    ]

    with open(output_csv_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Filename", "Transcription", "language"])

        with tqdm(total=len(audio_files), desc="Transcribing") as pbar:
            for file in audio_files:
                file_path = os.path.join(folder_path, file)
                transcription, language = transcribe_wav_to_text(file_path)
                csv_writer.writerow([file, transcription, language])
                pbar.update(1)  # Update progress bar

    print("Transcription completed. Output saved in", output_csv_file)


if __name__ == "__main__":
    main()
