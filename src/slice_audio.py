from pydub import AudioSegment
import os 
from tqdm import tqdm

def slice_and_save_audio(input_path, output_path, max_duration=5000):
    os.makedirs(output_path, exist_ok=True)

    for filename in tqdm(os.listdir(input_path)):
        if filename.endswith(".wav"):
            filepath = os.path.join(input_path, filename)
            audio = AudioSegment.from_file(filepath)

            if len(audio) > max_duration:
                segments = [audio[i: i+ max_duration] for i in range(0, len(audio), max_duration)]

                for i, segment in enumerate(segments):
                    output_filename = os.path.join(output_path, f"{os.path.splitext(filename)[0]}_{i+1}.wav")
                    segment.export(output_filename, format="wav")
            else:
                output_filename = os.path.join(output_path, filename)
                audio.export(output_filename, format="wav")


input_directory = '/home/auishik/noises/combined_noise_dataset'
output_directory = '/home/auishik/noises/combined_sliced_dataset'

slice_and_save_audio(input_directory, output_directory)