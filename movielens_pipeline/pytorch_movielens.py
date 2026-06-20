import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


# ========================================
# 1. HARDWARE & SEEDING CONFIG
# ========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ========================================
# 2. DATA LOADING & ID CLEANING
# ========================================
print("Loading MovieLens 100K data into Pandas...")
rating_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
df = pd.read_csv('./data/ml-100k/u.data', sep='\t', names=rating_columns)

# Real-world data IDs can be disconnected. Let's map them to strict 0-indexed positions.
# Example: user_id 1 -> index 0, user_id 2 -> index 1, etc.
df['user_idx'] = df['user_id'].astype('category').cat.codes
df['movie_idx'] = df['movie_id'].astype('category').cat.codes

num_users = df['user_idx'].nunique()
num_movies = df['movie_idx'].nunique()
print(f"Dataset completely re-mapped: {num_users} users, {num_movies} Movies")


# ========================================
# 3. THE PYTHON DATASTREAMING PIPELINE
# ========================================
class MovieLensDataset(Dataset):
    def __init__(self, dataframe):
        self.users = torch.tensor(dataframe['user_idx'].values, dtype=torch.long)
        self.movies = torch.tensor(dataframe['movie_idx'].values, dtype=torch.long)
        self.ratings = torch.tensor(dataframe['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

# Split data into Train (90%) and test (10%) sets
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(len(shuffled_df) * 0.9)
train_df = shuffled_df.iloc[:split_idx]
test_df = shuffled_df.iloc[split_idx:]

train_dataset = MovieLensDataset(train_df)
# DataLoader dynamically splits 90,000 training rows into streaming batches of 512
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)


# ========================================
# 4. NEURAL NETWORK ARCHITECTURE
# ========================================
class ScaledNCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(ScaledNCF, self).__init__()
        # Expanded embedding depth for real data patterns
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Deeper, wider neural layers to handle higher dimensions
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2) # Regularization to prevent overfitting

    def forward(self, user_indices, item_indices):
        user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding(item_indices)

        x = torch.cat([user_vec, item_vec], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        return self.output_layer(x).squeeze()


# Instantiate Model on to GPU
model = ScaledNCF(num_users, num_movies, embedding_dim=64).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # lower learning rate for precision


# ========================================
# 5. TRAINING LOOP OVER STREAMING BATCHES
# ========================================
print("\nTraining scaled recommendation engine accross real batches...")
epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    # Iterate through our streaming DataLoader batches
    for batch_users, batch_movies, batch_ratings in train_loader:
        # Move this specific batch chunk to the GPU
        batch_users = batch_users.to(device)
        batch_movies = batch_movies.to(device)
        batch_ratings = batch_ratings.to(device)

        # Optimize parameters
        predictions = model(batch_users, batch_movies)
        loss = criterion(predictions, batch_ratings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_users.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs} | Average Training Loss (MSE): {epoch_loss:.4f}")


# ========================================
# 6. SYSTEM EVALUATION (TESTING)
# ========================================
model.eval()
test_users = torch.tensor(test_df['user_idx'].values, dtype=torch.long).to(device)
test_movies = torch.tensor(test_df['movie_idx'].values, dtype=torch.long).to(device)
test_ratings = torch.tensor(test_df['rating'].values, dtype=torch.long).to(device)

with torch.no_grad():
    test_preds = model(test_users, test_movies)
    test_loss = criterion(test_preds, test_ratings)
    print(f"\nFinal Out-of-Sample Test Loss (MSE): {test_loss.item():.4f}")
    print(f"Calculated Error Margin (RMSE): {np.sqrt(test_loss.item()):.4f} stars")



# ========================================
# 7. EXPORT MODEL & MAPPINGS FOR PRODUCTION
# ========================================
print("\nExporting model weights and ID mappings for production API...")

# Save the PyTorch model weights
torch.save(model.state_dict(), './optimized_ncf_model.pth')

# Save our user and movie category codes so the API knows how to map real IDs
user_mapping = df.drop_duplicates('user_id').set_index('user_id')['user_idx'].to_dict()
movie_mapping = df.drop_duplicates('movie_id').set_index('movie_id')['movie_idx'].to_dict()

# Reverse mappings to convert internal indices back to real IDs for the final output
reverse_movie_mapping = {v: k for k, v in movie_mapping.items()}

import json
with open('./id_mappings.json', 'w') as f:
    json.dump({
        "user_mapping": user_mapping,
        "movie_mapping": movie_mapping,
        "reverse_movie_mapping": reverse_movie_mapping
    }, f)

print("Saved weights to 'optimized_ncf_model.pth' and mappings to 'id_mappings.json'!")
