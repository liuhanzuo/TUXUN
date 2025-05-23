import torch
import json
import re
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import math
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModel
from src.model import CountryModel
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from tqdm import tqdm

# 经纬度解析
def extract_lat_lon_from_text(answer) :
    # Find all matches of coordinates in parentheses
    matches = re.findall(r"\(([^)]+)\)", answer)
    
    if not matches:
        return None
    
    # Take the last match (most likely the actual answer)
    last_match = matches[-1]
    
    # Extract numbers from the last match
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

def extract_groundtruth_from_comment(info) -> Optional[Tuple[float, float]]:
    comments = info.get("Comment", [])
    lat, lon = None, None
    for entry in comments:
        if isinstance(entry, bytes):
            entry = entry.decode("utf-8", errors="ignore")
        lat_match = re.search(r"GPSLatitude=([-+]?\d+(\.\d+)?)", entry)
        lon_match = re.search(r"GPSLongitude=([-+]?\d+(\.\d+)?)", entry)
        if lat_match:
            lat = float(lat_match.group(1))
        if lon_match:
            lon = float(lon_match.group(1))
    if lat is not None and lon is not None:
        return lat, lon
    return None
def read_jpg_comment(file_path):
    try:
        with Image.open(file_path) as img:
            info = img.info
            print(info)
            if 'comment' in info:
                return info['comment']
            else:
                return "No Comment found"
    except Exception as e:
        return f'Error: {str(e)}'
# 加载模型
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf", device_map="auto")
model = AutoModelForImageTextToText.from_pretrained("llava-hf/llava-1.5-13b-hf", device_map="auto", torch_dtype=torch.bfloat16)
llava_device = "cuda:" + str(list(model.hf_device_map.values())[0])
vision_encoder = AutoModel.from_pretrained("openai/clip-vit-large-patch14-336").to(llava_device)
clip_image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
classifier = CountryModel(num_classes=214)
classifier.load_state_dict(torch.load("best_model.pth", map_location=llava_device))
classifier.to(llava_device)
classifier.eval()

with open("country2idx.json", "r") as f:
    country2idx = json.load(f)
idx2country = {v: k for k, v in country2idx.items()}

# 图像路径
from datasets import Dataset
val_dataset = Dataset.load_from_disk('./benchmark/yfcc')

distances = []
pred_coords_list = []
gt_coords_list = []
failed_images = []
valid_cnt = 0
pbar = tqdm(val_dataset, desc="Processing images", unit="img")
distance = [1, 25, 200, 750, 2500]
sum_dist = [0, 0, 0, 0, 0]
import io
for item in val_dataset:
    image = Image.open(io.BytesIO(item['img']))
    if image.mode != "RGB":
        image = image.convert("RGB")
    gt_coords = float(item['latitude']), float(item['longitude'])
    # 2. 提取 top-3 国家
    clip_inputs = clip_image_processor(images=image, return_tensors="pt").to(llava_device)
    with torch.no_grad():
        clip_output = vision_encoder.get_image_features(**clip_inputs)
        classifier_output = classifier(clip_output)
        top_probs, top_indices = torch.topk(classifier_output, k=5, dim=-1)
    top3_indices = top_indices[0][:3].tolist()
    top3_country_names = [idx2country[idx] for idx in top3_indices]
    # 3. 构造 prompt 并推理
    text_prompt = (
        "Image: <image>. Suppose you are an expert in geo-localization. "
        "Analyze this image and give your answer as a single pair of coordinates only. "
        "Do not include ranges or multiple values. Strictly output in this format: (latitude, longitude). "
        "No other text, only the coordinates in parentheses. "
        f"For your reference, the most probable country is: {top3_country_names[0]}. "
        "Your answer:"
    )
    inputs = processor(text=text_prompt, images=image, return_tensors="pt", padding=True, truncation=True).to(llava_device)
    output = model.generate(**inputs, max_new_tokens=64)
    answer = processor.decode(output[0], skip_special_tokens=True)
    pred_coords = extract_lat_lon_from_text(answer)
    if pred_coords is None:
        continue
    if pred_coords[0] == 0 or pred_coords[0] == 0:
        # print("shit")
        # print(answer)
        continue
    # 4. 计算误差
    # print(pred_coords, gt_coords)
    dist = haversine_distance(pred_coords[0], pred_coords[1], gt_coords[0], gt_coords[1])
    distances.append(dist)
    pred_coords_list.append(pred_coords)
    gt_coords_list.append(gt_coords)
    valid_cnt += 1
    for i in range(5):
        if dist < distance[i]:
            sum_dist[i] += 1
    # print(f" Pred={pred_coords}, GT={gt_coords}, Distance={dist:.2f} km, valid_cnt : {valid_cnt}")
    pbar.set_description(f"Pred={pred_coords}, GT={gt_coords}, Dist={dist:.2f}km, Valid={valid_cnt}, {sum_dist}")
    pbar.update(1)

# 统计结果
print(f"\nProcessed {len(distances)} images successfully.")
print(f"Average distance error: {np.mean(distances):.2f} km")
print(f"Median distance error: {np.median(distances):.2f} km")
print(f"Failed to process {len(failed_images)} images.")
for i in range(5):
    sum_dist[i] /= valid_cnt
print(f'proportion: {sum_dist}')

# 可视化误差分布
plt.hist(distances, bins=50, color='blue', edgecolor='black')
plt.title("Distribution of Localization Errors (km)")
plt.xlabel("Error Distance (km)")
plt.ylabel("Number of Images")
plt.grid(True)
plt.show()
