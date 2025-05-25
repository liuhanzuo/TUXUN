import os
from datasets import load_dataset, Dataset
from pathlib import Path
import json
import numpy as np
import faiss
import math
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModel
from PIL import Image
import io
import torch

# Create the output directory
output_dir = Path("./benchmark/im2gps3k")
output_dir.mkdir(parents=True, exist_ok=True)

with open("./benchmark/2k/image_gps_data.json", "r") as f:
    gps_data = json.load(f)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

samples = []
valid_count = 0
total_processed = 0

vision_encoder = AutoModel.from_pretrained("openai/clip-vit-large-patch14-336", device_map="auto")
clip_image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

dataset = Dataset.load_from_disk("./mp16_pro_with_clip_embeddings")
embeddings = np.vstack(dataset['embedding']).astype('float32')

dimension = embeddings.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)  # L2 distance index

index.add(embeddings)
print(f"FAISS index built with {index.ntotal} vectors.")

for example in gps_data:
    # Check if latitude and longitude exist and are not NaN
    gt_coords = [example['gps']['lat'], example['gps']['lon']]
    image = Image.open("./benchmark/" + example['file_path'])
    clip_inputs = clip_image_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        clip_output = vision_encoder.get_image_features(**clip_inputs)
        query_vector = clip_output.cpu().numpy()
        _ , indices = index.search(query_vector, 5)
    
    dataset_dist = haversine_distance(dataset['LAT'][indices[0][0]], dataset['LON'][indices[0][0]], float(gt_coords[0]), float(gt_coords[1]))

    if dataset_dist < 0.001: # make sure no test image is in training dataset
        total_processed += 1
        continue
    
    data = {'latitude': gt_coords[0], 'longitude': gt_coords[1]}
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    data['img'] = img_byte_arr.getvalue()
    samples.append(data)
    valid_count += 1
    total_processed += 1
    
    if valid_count % 100 == 0:  # Print progress every 100 valid samples
        print(f"Collected {valid_count} valid samples (processed {total_processed} total samples)...")

# Convert to regular Dataset
small_ds = Dataset.from_list(samples)

# Save the subset
save_path = output_dir
small_ds.save_to_disk(save_path)

print(f"\nSuccessfully saved {len(small_ds)} valid samples to {save_path}")
print(f"Total samples processed: {total_processed}")
print(f"Skipped {total_processed - valid_count} samples with invalid/missing coordinates")
