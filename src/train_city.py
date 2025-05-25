from model import CityModel

import torch
from datasets import Dataset
import os
import json
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score
from tqdm import tqdm
from torch import Tensor
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load country mapping
with open("shape_centers.json", "r") as f:
    shape_centers = json.load(f)

# Initialize model
num_classes = len(shape_centers)
print(f'Num Classes: {num_classes}')
model = CityModel(num_classes=num_classes, input_class=768)
model.to(device)

centers = []
for i in range(num_classes):
    centers.append(shape_centers[i]['center'])
center_tensor = Tensor(centers)
center_tensor = center_tensor.transpose(0, 1).to(device)

rad_torch = torch.tensor(6378137.0, dtype=torch.float64)

def haversine_matrix(x: Tensor, y: Tensor) -> Tensor:
    x_rad, y_rad = torch.deg2rad(x), torch.deg2rad(y)
    delta = x_rad.unsqueeze(2) - y_rad
    p = torch.cos(x_rad[:, 1]).unsqueeze(1) * torch.cos(y_rad[1, :]).unsqueeze(0)
    a = torch.sin(delta[:, 1, :] / 2)**2 + p * torch.sin(delta[:, 0, :] / 2)**2
    c = 2 * torch.arcsin(torch.sqrt(a))
    km = (rad_torch * c) / 1000
    return km

tau = 75

def HaversineLoss(probs, loc):
    distances = haversine_matrix(loc, center_tensor)
    minimum = distances.min(dim=-1, keepdim=True)
    distances = distances - minimum.values
    return (-probs.log() * (-distances / tau).exp()).sum(), minimum.indices

# Loss and optimizer
criterion = HaversineLoss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Load dataset
full_dataset = Dataset.load_from_disk("./mp16_pro_with_clip_embeddings")

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Custom collate function
def collate_fn(batch):
    batch = [b for b in batch if b is not None and b['embedding'] is not None]
    if len(batch) == 0:
        return None
    
    # Stack embeddings
    embeddings = torch.stack([torch.tensor(b['embedding']) for b in batch])
    lat = torch.stack([torch.tensor(b['LAT']) for b in batch])
    lon = torch.stack([torch.tensor(b['LON']) for b in batch])
    
    # Convert country names to indices and create one-hot vectors
    return {
        'embedding': embeddings,
        'loc': torch.stack([lat, lon], dim=-1)
    }

# Create DataLoaders
train_dataloader = DataLoader(
    train_dataset,
    batch_size=1024,
    shuffle=True,
    collate_fn=collate_fn
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=1024,
    shuffle=False,
    collate_fn=collate_fn
)

def evaluate(model, dataloader):
    model.eval()
    val_loss = 0
    val_acc = 0
    processed_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
                
            images = batch['embedding'].to(device)
            loc = batch['loc'].to(device)
            
            outputs = model(images, return_logits=False)
            loss, label = criterion(outputs, loc)
            val_acc += (outputs.max(dim=-1).indices == label.reshape(-1)).sum().item()
            processed_samples += len(loc)
            val_loss += loss.item() 
    
    return val_acc / processed_samples

# Training loop
num_epochs = 20
best_val_loss = float('inf')
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=num_epochs,
    eta_min=1e-5
)

best_val_acc = 0
for epoch in tqdm(range(num_epochs)):
    # Training phase
    model.train()
    total_loss = 0
    train_acc = 0
    processed_samples = 0
    
    for batch in tqdm(train_dataloader):
        if batch is None:
            continue
            
        images = batch['embedding'].to(device)
        loc = batch['loc'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images, return_logits=False)
        loss, label = criterion(outputs, loc)
        train_acc += (outputs.max(dim=-1).indices == label.reshape(-1)).sum().item()

        loss.backward()
        optimizer.step()
        
        batch_size = len(loc)
        total_loss += loss.item() 
        processed_samples += batch_size
    
    scheduler.step()
    # Print training stats
    if processed_samples > 0:
        avg_train_loss = total_loss / processed_samples
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Processed samples: {processed_samples}")
        print(f"Accuracy: {train_acc / processed_samples:.4f}")
        print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")
    else:
        print(f"\nEpoch {epoch+1}/{num_epochs}: No training samples processed")
        continue
    
    # Validation phase
    val_acc = evaluate(model, val_dataloader)
    
    print(f"Val Metrics:")
    print(f"Accuracy: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "./city_siglip.pth")
        print("Saved new best model")
