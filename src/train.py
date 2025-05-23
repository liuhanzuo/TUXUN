from model import CountryModel

import torch
from datasets import Dataset
import os
import json
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score
from tqdm import tqdm
# Prepare device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
num_classes = 214
model = CountryModel(num_classes=num_classes)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Load dataset
full_dataset = Dataset.load_from_disk("/scorpio/home/liuhanzuo/Img2Loc/mp16_pro_with_clip_embeddings")

# Split into train and validation sets (80-20 split)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Load country mapping
with open("country2idx.json", "r") as f:
    country_idx_to_name = json.load(f)
country_name_to_idx = {k: int(v) for k, v in country_idx_to_name.items()}

# Verify all labels are within bounds
max_idx = max(country_name_to_idx.values())
min_idx = min(country_name_to_idx.values())
print(f"Label range: {min_idx} to {max_idx}")
assert max_idx < num_classes, f"Max label index {max_idx} >= num_classes {num_classes}"
assert min_idx >= 0, f"Min label index {min_idx} is negative"

# Custom collate function
def collate_fn(batch):
    batch = [b for b in batch if b is not None and b['embedding'] is not None]
    if len(batch) == 0:
        return None
    
    # Stack embeddings
    embeddings = torch.stack([torch.tensor(b['embedding']) for b in batch])
    
    # Convert country names to indices and create one-hot vectors
    labels = [country_name_to_idx[b['country']] for b in batch]
    one_hot_labels = torch.zeros(len(labels), num_classes)
    one_hot_labels[range(len(labels)), labels] = 1
    
    return {
        'embedding': embeddings,
        'country': torch.tensor(labels).long(),
        'one_hot_labels': one_hot_labels
    }

# Create DataLoaders
train_dataloader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn
)

def evaluate(model, dataloader):
    model.eval()
    all_probs = []
    all_labels = []
    all_one_hot = []
    val_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
                
            images = batch['embedding'].to(device)
            labels = batch['country'].to(device)
            one_hot_labels = batch['one_hot_labels'].to(device)
            
            outputs = model(images, return_logits=True)
            probs = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * len(labels)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_one_hot.append(one_hot_labels.cpu().numpy())
    
    if len(all_labels) == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), None
    
    # Concatenate all batches
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_one_hot = np.concatenate(all_one_hot)
    
    avg_loss = val_loss / len(all_labels)
    
    # Calculate metrics without using argmax
    # 1. Probability-based accuracy (using ground truth class probability)
    prob_acc = np.mean(all_probs[range(len(all_labels)), all_labels])
    
    # 2. Cross-entropy between predictions and one-hot labels
    ce_loss = -np.mean(np.log(all_probs[all_one_hot == 1] + 1e-10))
    
    # Get all possible class labels from the mapping
    all_possible_labels = np.array(list(country_name_to_idx.values()))
    
    # 3. Top-k accuracy (using probability distribution)
    try:
        top3_acc = top_k_accuracy_score(
            all_labels, 
            all_probs, 
            k=3,
            labels=all_possible_labels  # Provide all possible classes
        )
    except ValueError:
        # Fallback if there's still an issue
        unique_labels = np.unique(all_labels)
        top3_acc = top_k_accuracy_score(
            all_labels,
            all_probs[:, unique_labels],  # Only use columns for present classes
            k=min(3, len(unique_labels))
        )
    
    # Generate classification report
    pred_classes = np.argmax(all_probs, axis=1)
    report = classification_report(
    all_labels,
    pred_classes,
    labels=np.unique(all_labels),
    target_names=[country for country, idx in country_name_to_idx.items() 
                 if idx in np.unique(all_labels)],
    zero_division=0  
)

    
    return avg_loss, prob_acc, top3_acc, ce_loss, report


# Training loop
num_epochs = 30
best_val_loss = float('inf')
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=num_epochs,      # 训练总轮数
    eta_min=1e-5           # 最小学习率
)
for epoch in tqdm(range(num_epochs)):
    # Training phase
    model.train()
    total_loss = 0
    processed_samples = 0
    
    for batch in tqdm(train_dataloader):
        if batch is None:
            continue
            
        images = batch['embedding'].to(device)
        labels = batch['country'].to(device)
        one_hot_labels = batch['one_hot_labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images, return_logits = True)
        # print(outputs)
        # print(outputs.shape)
        # print(labels)
        # assert False
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        batch_size = len(labels)
        total_loss += loss.item() * batch_size
        processed_samples += batch_size
    scheduler.step()
    # Print training stats
    if processed_samples > 0:
        avg_train_loss = total_loss / processed_samples
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Processed samples: {processed_samples}")
        print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")
    else:
        print(f"\nEpoch {epoch+1}/{num_epochs}: No training samples processed")
        continue
    
    # Validation phase
    val_loss, prob_acc, top3_acc, ce_loss, val_report = evaluate(model, val_dataloader)
    print(f"Val Metrics:")
    print(f"- Loss: {val_loss:.4f}")
    print(f"- Prob Accuracy: {prob_acc:.4f} (average ground truth class probability)")
    print(f"- Top-3 Accuracy: {top3_acc:.4f}")
    print(f"- Cross-Entropy: {ce_loss:.4f} (direct prob vs one-hot)")
    
    # Print classification report (first 20 classes for readability)
    if val_report:
        print("\nValidation Classification Report (first 20 classes):")
        lines = val_report.split('\n')[:22]  # Header + 20 classes
        print('\n'.join(lines))
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved new best model")
