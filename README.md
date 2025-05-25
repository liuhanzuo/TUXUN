Ablation Study
VLM w geocell w country [85, 259, 622, 1081, 1457], [3, 153, 634, 1158, 1495] ,[134, 334, 496, 853, 1289] 2k
VLM o geocell o country [86, 247, 586, 1029, 1410], [3, 153, 634, 1158, 1495] ,[134, 334, 496, 853, 1289] 2k

# TUXUN

## Installation

Instructions on how to install and set up the project.

```bash
# Clone the repository
git clone git@github.com:liuhanzuo/TUXUN.git

# Change to the project directory
cd TUXUN

# Create a conda environment
conda create -n tuxun python=3.10

# Activate the conda environment
conda activate tuxun

# Install the project dependencies
pip install -r requirements.txt

# Install the faiss database
conda install faiss-gpu
```

## Prepare dataset
```bash
# Download 2k_random_test from im2gps3k dataset and store it in ./benchmark/2k
# Preprocess the im2gps3k evaluation dataset
python src/im2gps3k_data_preprocess.py

# Prepare the YFCC4k evaluation dataset
python src/yfcc4k_data_preprocess.py
```

## Achieve CLIP Embeddings and Train Classifiers

```bash
# Load the dataset and generate CLIP Embeddings
python src/embeddings.python

# Train the country classifier
python src/train_country.py

# Train the geocell classifier
python src/train_city.py
```

## Usage
```bash
# Run the main code!
python main.py
```