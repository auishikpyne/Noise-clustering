import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load the embeddings from the text file
input_file_path = '/home/auishik/noise_clustering/saved_files/embeddings.txt'
embeddings_array = np.loadtxt(input_file_path, delimiter='\t')
print(embeddings_array)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings_array)

# Plot the results
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1])
plt.title('t-SNE Visualization of Embeddings')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

# Save the plot as an image
output_image_path = '/home/auishik/noise_clustering/saved_files/tsne_embeddings_plot.png'
plt.savefig(output_image_path)

print(f"t-SNE embeddings plot saved to {output_image_path}")
