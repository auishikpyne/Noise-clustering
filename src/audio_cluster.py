import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from tqdm import tqdm
import glob

# Load the embeddings from the text file
input_file_path = '/home/auishik/noise_clustering/saved_files/embeddings.txt'
embeddings_array = np.loadtxt(input_file_path, delimiter='\t')

# Load the audio file paths
audio_files = glob.glob('/home/auishik/noise_clustering/noise_train/*.wav')


# Apply t-SNE
tsne = TSNE(n_components=3, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings_array)

# Number of clusters to try
num_clusters = 14

# Apply KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings_tsne)

# Create a dictionary to map cluster labels to audio files
cluster_to_files = {cluster_label: [] for cluster_label in np.unique(cluster_labels)}

# Populate the dictionary
for file, cluster_label in zip(audio_files, cluster_labels):
    cluster_to_files[cluster_label].append(file)

# Print audio files for each cluster
for cluster_label, files in cluster_to_files.items():
    print(f"Audio files in Cluster {cluster_label}:")
    for file in files:
        print(file)
    print("\n")
    

