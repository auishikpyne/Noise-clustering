import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the embeddings from the text file
input_file_path = '/home/auishik/noise_clustering/saved_files/embeddings.txt'
embeddings_array = np.loadtxt(input_file_path, delimiter='\t')

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings_array)

# Apply KMeans clustering with k=12
kmeans = KMeans(n_clusters=27, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings_tsne)
print(cluster_labels)
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
plt.title('t-SNE Visualization of Embeddings with Clusters')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(label='Cluster Label')

output_image_path = '/home/auishik/noise_clustering/saved_files/tsne_embeddings_cluster_plot.png'
plt.savefig(output_image_path)

print(f"t-SNE embeddings plot with clusters saved to {output_image_path}")
