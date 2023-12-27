import glob
from tqdm import tqdm
import numpy as np
import nemo.collections.asr as nemo_asr
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from multiprocessing import Pool
import multiprocessing as mp
import os
import shutil
import json


# Set the start method to 'spawn'
mp.set_start_method('spawn', force=True)

# Load the speaker model
speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large", map_location='cpu')


if __name__ == "__main__":
    
    audio_files = glob.glob('/home/auishik/noises/combined_sliced_dataset/*.wav')

    print(len(audio_files))
    
    embeddings_list = []

    for file in tqdm(audio_files):
        try:
            embedding = speaker_model.get_embedding(file)
            embeddings_list.append(embedding.numpy())
        except Exception as e:
            # Handle the exception (you can print an error message, log it, or take other actions)
            print(f"Error processing file {file}: {e}")

    # Convert the list of embeddings to a NumPy array
    embeddings_array = np.vstack(embeddings_list)


    # Specify the path to the text file
    output_file_path = '/home/auishik/noise_clustering/saved_files/embeddings.txt'

    os.remove(output_file_path)
    # Save the embeddings to a text file
    np.savetxt(output_file_path, embeddings_array, delimiter='\t', fmt='%f')

    print(f"Embeddings saved to {output_file_path}")
    