import faiss
import numpy as np
import os
from datasets import Dataset
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModel
import torch

# Path to your dataset
dataset_path = './mp16_pro_with_clip_embeddings'

# Load the embeddings using the dataset's save_to_disk method
dataset = Dataset.load_from_disk(dataset_path)

# Extract embeddings from the dataset and ensure they are of type float32
embeddings = np.vstack(dataset['embedding']).astype('float32')

# Create a FAISS index
dimension = embeddings.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)  # L2 distance index

# Add embeddings to the index
index.add(embeddings)

print(f"FAISS index built with {index.ntotal} vectors.")

vision_encoder = AutoModel.from_pretrained("openai/clip-vit-large-patch14-336")
clip_image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

image = Image.open('image_boston.jpg')
clip_inputs = clip_image_processor(images=image, return_tensors="pt")
with torch.no_grad():
    clip_output = vision_encoder.get_image_features(**clip_inputs)

# Example query vector (ensure it is of type float32 and has the same dimension as the index)
query_vector = clip_output.numpy()

# Perform a search on the index
k = 5  # Number of nearest neighbors to retrieve
distances, indices = index.search(query_vector, k)

# Print the results
print(f"Query vector: {query_vector}")
print(f"Nearest neighbors' indices: {indices}")
# Assuming the dataset has a 'country' column corresponding to each embedding
nearest_countries = [dataset['country'][i] for i in indices[0]]
print(f"Distances to nearest neighbors: {distances}")
print(f"Countries of nearest neighbors: {nearest_countries}")
print(f"Nearest location: {dataset['LAT'][indices[0][0]]}, {dataset['LON'][indices[0][0]]}")