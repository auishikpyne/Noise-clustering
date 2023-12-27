from pydub import AudioSegment
import os

def print_audio_durations(directory):
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith((".mp3", ".wav")):
            audio_file_path = os.path.join(directory, filename)
            
            # Load the audio file
            audio = AudioSegment.from_file(audio_file_path)

            # Calculate the duration in seconds
            duration_seconds = len(audio) / 1000

            print(f"{filename}: {duration_seconds:.2f} seconds")

if __name__ == "__main__":
    # Replace 'your_directory' with the path to the directory containing the audio files
    print_audio_durations('/home/auishik/noises/combined_noise_dataset')
