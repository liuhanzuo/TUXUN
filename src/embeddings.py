from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset, Features, Value
import torch
from PIL import Image
import numpy as np

# Load model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Load dataset
dataset = load_dataset("Jia-py/MP16-Pro")

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def extract_embeddings(batch):
    # Process images
    images = []
    for img_id in batch['IMG_ID']:
        img = Image.open(f"./images/{img_id}")
        if img.mode != 'RGB':
            img = img.convert('RGB')
        images.append(img)
    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get image embeddings
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    # Convert to numpy and return
    return {'embedding': outputs.cpu().numpy()}

# Map the function to the dataset
print(dataset)
embedded_dataset = dataset['train'].select(range(100000)).map(
    extract_embeddings,
    batched=True,
    batch_size=1024,  # Adjust based on your GPU memory
)
print(embedded_dataset)

# The embeddings are now stored in the 'embedding' column
print(np.array(embedded_dataset[0]['embedding']).shape)

# Save the embeddings if needed
embedded_dataset.save_to_disk("./mp16_pro_with_clip_embeddings")