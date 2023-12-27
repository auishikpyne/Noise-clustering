from pydub import AudioSegment
import os
from tqdm import tqdm

def convert_mp3_to_wav(mp3_file):
    # Load the MP3 file
    sound = AudioSegment.from_mp3(mp3_file)

    # Construct the output WAV file name
    wav_file = os.path.splitext(mp3_file)[0] + '.wav'

    # Save the WAV file
    sound.export(wav_file, format="wav")

    # Optional: Remove the original MP3 file if you want to replace it
    os.remove(mp3_file)

def convert_all_mp3_to_wav(directory):
    # Loop through all files in the directory
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".mp3"):
            mp3_file = os.path.join(directory, filename)
            convert_mp3_to_wav(mp3_file)


if __name__ == "__main__":
    # Replace 'your_directory' with the path to the directory containing the .mp3 files
    convert_all_mp3_to_wav('/home/auishik/noises/combined_noise_dataset')

