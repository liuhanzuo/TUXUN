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
output_dir = Path("./benchmark/yfcc")
output_dir.mkdir(parents=True, exist_ok=True)

# Load dataset with streaming
ds = load_dataset("dalle-mini/YFCC100M_OpenAI_subset", 
                 trust_remote_code=True,
                 split='train',
                 streaming=True)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c
# Collect valid samples
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
for example in ds:
    if valid_count >= 4000:
        break
    
    # Check if latitude and longitude exist and are not NaN
    if ('latitude' in example and 'longitude' in example and not example['latitude']=='nan' and not example['longitude'] == 'nan'):
        gt_coords = [example['latitude'], example['longitude']]
        image = Image.open(io.BytesIO(example['img']))
        clip_inputs = clip_image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            clip_output = vision_encoder.get_image_features(**clip_inputs)
            query_vector = clip_output.cpu().numpy()
            _ , indices = index.search(query_vector, 5)
        # print(gt_coords)
        # print(int(gt_coords[0]),int(gt_coords[1]))
        dataset_dist = haversine_distance(dataset['LAT'][indices[0][0]], dataset['LON'][indices[0][0]], float(gt_coords[0]), float(gt_coords[1]))
        print(dataset['LAT'][indices[0][0]], dataset['LON'][indices[0][0]], float(gt_coords[0]), float(gt_coords[1]), dataset_dist)
        if dataset_dist < 0.001:
            print(f'Sample picture, continue')
            total_processed += 1
            continue
        samples.append(example)
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

# import math
# import torch
# import io
# import faiss
# import numpy as np
# from PIL import Image
# from datasets import Dataset
# from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModel

# vision_encoder = AutoModel.from_pretrained("openai/clip-vit-large-patch14-336")
# clip_image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

# val_dataset = Dataset.load_from_disk('./benchmark/yfcc')
# dataset = Dataset.load_from_disk("./mp16_pro_with_clip_embeddings")
# embeddings = np.vstack(dataset['embedding']).astype('float32')

# dimension = embeddings.shape[1]  # Dimension of the embeddings
# index = faiss.IndexFlatL2(dimension)  # L2 distance index

# index.add(embeddings)
# print(f"FAISS index built with {index.ntotal} vectors.")


# tot = 0
# for item in val_dataset:
#     image = Image.open(io.BytesIO(item['img']))
#     if image.mode != "RGB":
#         image = image.convert("RGB")
#     gt_coords = float(item['latitude']), float(item['longitude'])
#     clip_inputs = clip_image_processor(images=image, return_tensors="pt")
#     with torch.no_grad():
#         clip_output = vision_encoder.get_image_features(**clip_inputs)
#         query_vector = clip_output.cpu().numpy()
#         _ , indices = index.search(query_vector, 5)
#     dataset_dist = haversine_distance(dataset['LAT'][indices[0][0]], dataset['LON'][indices[0][0]], gt_coords[0], gt_coords[1])
#     if dataset_dist < 0.001:
#         tot += 1
#         print(f"distance: {dataset_dist}, {dataset['IMG_ID'][indices[0][0]]}, {tot}")
#         # image.save(f"./test_images/{tot}.jpg")