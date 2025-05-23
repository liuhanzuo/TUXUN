# import os
# from datasets import load_dataset, Dataset
# from pathlib import Path
# import json
# import numpy as np

# # Create the output directory
# output_dir = Path("./benchmark/yfcc")
# output_dir.mkdir(parents=True, exist_ok=True)

# # Load dataset with streaming
# ds = load_dataset("dalle-mini/YFCC100M_OpenAI_subset", 
#                  trust_remote_code=True,
#                  split='train',
#                  streaming=True)

# # Collect valid samples
# samples = []
# valid_count = 0
# total_processed = 0

# for example in ds:
#     if valid_count >= 4000:
#         break
        
#     # Check if latitude and longitude exist and are not NaN
#     if ('latitude' in example and 'longitude' in example and not example['latitude']=='nan' and not example['longitude'] == 'nan'):
        
#         samples.append(example)
#         valid_count += 1
        
#     total_processed += 1
    
#     if valid_count % 100 == 0:  # Print progress every 100 valid samples
#         print(f"Collected {valid_count} valid samples (processed {total_processed} total samples)...")

# # Convert to regular Dataset
# small_ds = Dataset.from_list(samples)

# # Save the subset
# save_path = output_dir
# small_ds.save_to_disk(save_path)

# print(f"\nSuccessfully saved {len(small_ds)} valid samples to {save_path}")
# print(f"Total samples processed: {total_processed}")
# print(f"Skipped {total_processed - valid_count} samples with invalid/missing coordinates")

a = [0, 1, 2]
print(a/3)