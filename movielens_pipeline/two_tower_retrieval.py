import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

# Setup hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


# ========================================
# 1. LOAD DATA & SIMULATE POSITIVE INTERACTIONS 
# ========================================
# In retrieval, we want to predict IF a user will watch a movie (Binary/Implicit feedback)
rating_columns = ["user_id", "movie_id", "rating", "timestaamp"]
df = pd.read_csv("./data/ml-100k/u.data", sep='\t', names=rating_columns)

# Map continous codes
df['user_idx'] = df['user_id'].astype('category').cat.codes
df['movie_idx'] = df['movie_id'].astype('category').cat.codes

num_users = df['user_idx'].nunique()
num_movies = df['movie_idx'].nunique()

# Filter for "positive interactions" (movies the user likes, rated 4 or 5 stars)
positive_pairs = df[df['rating'] >= 4][['user_idx', 'movie_idx']].values


# ========================================
# 2. DEFINE THE TWO TOWERS
# ========================================
class UserTower(nn.Module):
    def __init__(self, num_users, embedding_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_users, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 16) # Compress to a final space
    
    def forward(self, user_ids):
        x = self.embedding(user_ids)
        return F.normalize(self.fc(x), p=2, dim=1)

class ItemTower(nn.Module):
    def __init__(self, num_movies, embedding_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_movies, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 16)
    
    def forward(self, movie_ids):
        x = self.embedding(movie_ids)
        return F.normalize(self.fc(x), p=2, dim=1) # Normalize vectors onto a unit hypersphere



# ========================================
# 3. TRAINING WITH CONTRASTIVE LOSS
# ========================================
user_tower = UserTower(num_users).to(device)
item_tower = ItemTower(num_movies).to(device)

# Optimize both towers together
optimizer = torch.optim.Adam(list(user_tower.parameters()) + list(item_tower.parameters()), lr=0.001)

print("Training Two-Tower Retrieval System (Candidate Generation)...")
epochs = 3
batch_size = 256

for epoch in range(epochs):
    # Shuffle positive pairs
    np.random.shuffle(positive_pairs)
    runnning_loss = 0.0

    for i in range(0, len(positive_pairs), batch_size):
        batch = positive_pairs[i:i+batch_size]
        if len(batch) < batch_size: break

        u_batch = torch.tensor(batch[:, 0], dtype=torch.long).to(device)
        m_batch = torch.tensor(batch[:, 1], dtype=torch.long).to(device)

        # Pass data through respective isolated towers
        user_embeddings = user_tower(u_batch)
        item_embeddings = item_tower(m_batch)

        # Calculate Dot Product Similarity matrix between all users and item in the batch
        # This outputs a [256, 256] grid of similarity scores
        similarity_matrix = torch.matmul(user_embeddings, item_embeddings.T)

        # Ground Truth: The Diagonal elements are the true watched pairs (Positive targets)
        # All other elements off the diagonal act as "In-Batch Negative samples"
        targets = torch.arange(batch_size).to(device)

        # CrossEntropyLoss automatically forces the diagonal similarity scores higher 
        # and pushes the off_diagonal incorrect pairs down to 0
        loss = F.cross_entropy(similarity_matrix, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        runnning_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Retriveal Contrastive Loss: {runnning_loss / (len(positive_pairs)//batch_size):.4f}")

print("\nTwo-Tower Pipeline successfully trained!") 
