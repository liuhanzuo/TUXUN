import torch
import json
import re
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import math
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModel
from src.model import CountryModel, CityModel
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from tqdm import tqdm
import faiss
from datasets import Dataset
import io

def extract_lat_lon_from_text(answer):
    matches = re.findall(r"\(([^)]+)\)", answer)
    
    if not matches:
        return None
    
    last_match = matches[-1]
    
    numbers = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", last_match)
    
    if len(numbers) >= 2:
        try:
            lat = float(numbers[0])
            lon = float(numbers[1])
            return lat, lon
        except ValueError:
            return None
    
    return None

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf", device_map="auto")
model = AutoModelForImageTextToText.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf", device_map="auto", torch_dtype=torch.bfloat16)
llava_device = "cuda:" + str(list(model.hf_device_map.values())[0])
vision_encoder = AutoModel.from_pretrained("openai/clip-vit-large-patch14-336").to(llava_device)
clip_image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

classifier = CountryModel(num_classes=214)
classifier.load_state_dict(torch.load("model/clip_large.pth", map_location=llava_device))
classifier.to(llava_device)
classifier.eval()

city_classifier = CityModel(num_classes=3224)
city_classifier.load_state_dict(torch.load("model/city.pth", map_location=llava_device))
city_classifier.to(llava_device)
city_classifier.eval()

with open("country2idx.json", "r") as f:
    country2idx = json.load(f)
idx2country = {v: k for k, v in country2idx.items()}
with open("shape_centers.json", "r") as f:
    shape_cneter = json.load(f)

val_dataset = Dataset.load_from_disk('./benchmark/yfcc')
dataset = Dataset.load_from_disk("./mp16_pro_with_clip_embeddings")
embeddings = np.vstack(dataset['embedding']).astype('float32')

dimension = embeddings.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)  # L2 distance index

index.add(embeddings)
print(f"FAISS index built with {index.ntotal} vectors.")

distances = []
geocell_distances = []
dataset_distances = []
pred_coords_list = []
gt_coords_list = []
valid_cnt = 0

pbar = tqdm(val_dataset, desc="Processing images", unit="img")

distance = [1, 25, 200, 750, 2500]
sum_dist = [0, 0, 0, 0, 0]
geocell_sum_dist = [0, 0, 0, 0, 0]
dataset_sum_dist = [0, 0, 0, 0, 0]

for item in val_dataset:
    image = Image.open(io.BytesIO(item['img']))
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    gt_coords = float(item['latitude']), float(item['longitude'])
    clip_inputs = clip_image_processor(images=image, return_tensors="pt").to(llava_device)
    with torch.no_grad():
        clip_output = vision_encoder.get_image_features(**clip_inputs)
        classifier_output = classifier(clip_output)
        top_probs, top_indices = torch.topk(classifier_output, k=5, dim=-1)
        city_output = city_classifier(clip_output)
        city_top_probs, city_top_indices = torch.topk(city_output, k=5, dim=-1)
        query_vector = clip_output.cpu().numpy()
        database_dis , indices = index.search(query_vector, 10)
        nearest_countries = [dataset['country'][i] for i in indices[0]]
    
    top3_indices = top_indices[0][:3].tolist()
    top3_country_names = [idx2country[idx] for idx in top3_indices]
    top3_city_indices = city_top_indices[0][:3].tolist()
    top3_city_names = [shape_cneter[idx] for idx in top3_city_indices]

    if database_dis[0][9] - database_dis[0][0] > 40.0:
        pred_coords = [dataset['LAT'][indices[0][0]], dataset['LON'][indices[0][0]]]
    else:
        text_prompt = (
            "Image: <image>. Suppose you are an expert in geo-localization. "
            "Analyze this image and give your answer as a single pair of coordinates only. "
            "Do not include ranges or multiple values. Strictly output in this format: (latitude, longitude). "
            "No other text, only the coordinates in parentheses. "
            # f"For your reference, the nearest city is {top3_city_names[0]['shapeName']} in {top3_country_names[0]}. "
            "Your answer:"
        )
        inputs = processor(text=text_prompt, images=image, return_tensors="pt", padding=True, truncation=True).to(llava_device)
        output = model.generate(**inputs, max_new_tokens=64)
        answer = processor.decode(output[0], skip_special_tokens=True)
        pred_coords = extract_lat_lon_from_text(answer)
        if pred_coords is None:
            pred_coords = top3_city_names[0]['center']
        elif pred_coords[0] == 0 or pred_coords[0] == 0:
            pred_coords = top3_city_names[0]['center']

    dist = haversine_distance(pred_coords[0], pred_coords[1], gt_coords[0], gt_coords[1])
    geocell_dist = haversine_distance(top3_city_names[0]['center'][0], top3_city_names[0]['center'][1], gt_coords[0], gt_coords[1])
    dataset_dist = haversine_distance(dataset['LAT'][indices[0][0]], dataset['LON'][indices[0][0]], gt_coords[0], gt_coords[1])
    
    distances.append(dist)
    geocell_distances.append(geocell_dist)
    dataset_distances.append(dataset_dist)
    pred_coords_list.append(pred_coords)
    gt_coords_list.append(gt_coords)
    
    valid_cnt += 1
    for i in range(5):
        if dist < distance[i]:
            sum_dist[i] += 1
        if geocell_dist < distance[i]:
            geocell_sum_dist[i] += 1
        if dataset_dist < distance[i]:
            dataset_sum_dist[i] += 1
    pbar.set_description(f"Pred={pred_coords}, GT={gt_coords}, Dist={dist:.2f}km, Valid={valid_cnt}, {sum_dist}, {geocell_sum_dist} ,{dataset_sum_dist}")
    pbar.update(1)

print(f"\nProcessed {len(distances)} images successfully.")
print(f"Average distance error: {np.mean(distances):.2f}, {np.mean(geocell_distances)}, {np.mean(dataset_distances)} km")

for i in range(5):
    sum_dist[i] /= valid_cnt
    geocell_sum_dist[i] /= valid_cnt
    dataset_sum_dist[i] /= valid_cnt
print(f'proportion: {sum_dist}, {geocell_sum_dist}, {dataset_sum_dist}')


