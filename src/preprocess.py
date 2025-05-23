from model import CountryModel
from transformers import CLIPVisionModel, CLIPImageProcessor
import torch
from datasets import Dataset
import os
import json
import torch.nn as nn

dataset = Dataset.load_from_disk("/scorpio/home/liuhanzuo/Img2Loc/mp16_pro_with_clip_embeddings")
country = set()
for item in dataset:
    if "country" in item:
        country.add(item["country"])
        # if item["country_code"] not in country:
        #     country[item["country_code"]] = item["country"]
        # else:
        #     assert country[item["country_code"]] == item["country"], f"Country code mismatch for {item['country_code'], item['country']} and {country[item['country_code']]}"
print(list(country))
print(len(country))
country2idx = {c: idx for idx, c in enumerate(sorted(country))}
with open("country2idx.json", "w") as f:
    json.dump(country2idx, f, indent=2)
