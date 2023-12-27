import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
# Load the embeddings from the text file
input_file_path = '/home/auishik/noise_clustering/saved_files/embeddings.txt'
embeddings_array = np.loadtxt(input_file_path, delimiter='\t')

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings_array)

# Range of cluster values to try
cluster_range = range(1500, 2500, 10)

# Lists to store silhouette scores
silhouette_scores = []

# Loop through different numbers of clusters
for num_clusters in tqdm(cluster_range):
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=int(num_clusters), random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_tsne)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(embeddings_tsne, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot silhouette scores for different numbers of clusters
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(np.arange(int(np.min(cluster_range)), int(np.max(cluster_range)) + 1, 100))

output_image_path = '/home/auishik/noise_clustering/saved_files/tsne_Silhouette_plot_2d_5.png'
plt.savefig(output_image_path)
plt.show()
