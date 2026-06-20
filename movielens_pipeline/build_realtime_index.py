import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import faiss
import json

# Force to CPU for consistent index extraction
device = torch.device("cpu")

# 1. LOAD DATA DEFINITIONS WITH CORRECT PATHS
print("Loading MovieLens dataset tables...")
rating_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
df = pd.read_csv('./data/ml-100k/u.data', sep='\t', names=rating_columns)

# Cleanly map category codes to align with training
df['movie_idx'] = df['movie_id'].astype('category').cat.codes
num_movies = df['movie_idx'].nunique()

# Load the true movie titles
movie_info_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + [f'genre_{i}' for i in range(19)]
movies_df = pd.read_csv('./data/ml-100k/u.item', sep='|', names=movie_info_cols, encoding='latin-1')

# Create an exact lookup dictionary: movie_id -> title string
movie_id_to_title = movies_df.set_index('movie_id')['title'].to_dict()

# Create clean mappings between internal matrix index and real movie text titles
movie_idx_to_title = {}
movie_id_map = df.drop_duplicates('movie_id').set_index('movie_id')['movie_idx'].to_dict()

for m_id, m_idx in movie_id_map.items():
    # Fetch the real text name using the movie_id
    real_name = movie_id_to_title.get(m_id, f"Unknown Movie {m_id}")
    movie_idx_to_title[int(m_idx)] = real_name

# 2. MATCH THE EXACT ARCHITECTURE WE TRAINED
class OptimizedNCF(nn.Module):
    def __init__(self, num_users=943, num_movies=1682, embedding_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_movies, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, user_indices, item_indices):
        user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding(item_indices)
        x = torch.cat([user_vec, item_vec], dim=-1)
        x = self.relu(self.fc1(x))
        return self.output_layer(x).squeeze()

# Load our saved optimized model weights
print("Loading trained weights from optimized_ncf_model.pth...")
full_model = OptimizedNCF()
full_model.load_state_dict(torch.load('./optimized_ncf_model.pth', map_location=device))
full_model.eval()

# 3. EXTRACT GENUINE HIGH-PERFORMANCE VECTORS
print("Extracting item embedding vectors from our trained model...")
with torch.no_grad():
    # Pull the exact movie embedding weights learned during training
    raw_vectors = full_model.item_embedding.weight.data.numpy().astype('float32')

# 4. EXPORT TO FAISS
print(f"Building Faiss index with {raw_vectors.shape[0]} movies across {raw_vectors.shape[1]} dimensions...")
index = faiss.IndexFlatL2(64) # Crucial: Embedding dimension is 64 now!
index.add(raw_vectors)

faiss.write_index(index, "./realtime_movies.index")

# Save movie index-to-title mappings securely
with open("./realtime_mappings.json", "w") as f:
    json.dump({
        "movie_idx_to_title": movie_idx_to_title,
        "movie_title_to_idx": {v: k for k, v in movie_idx_to_title.items()}
    }, f)

print("\nSuccess! Real-time catalog re-mapped and exported cleanly.")

