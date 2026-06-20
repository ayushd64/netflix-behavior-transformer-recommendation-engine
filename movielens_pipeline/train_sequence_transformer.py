import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

# =========================================================================
# 1. DATA PREPROCESSING & SEQUENCE EXTRACTION
# =========================================================================
print("Loading MovieLens dataset for sequence transformations...")
rating_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
df = pd.read_csv('./data/ml-100k/u.data', sep='\t', names=rating_columns)

# Cleanly map category codes to ensure clean index tracking
df['movie_idx'] = df['movie_id'].astype('category').cat.codes
num_movies = df['movie_idx'].nunique()

# Group by user and sort strictly by historical interaction timeline
print("Building chronological user watch histories...")
user_histories = df.sort_values('timestamp').groupby('user_id')['movie_idx'].apply(list).to_dict()

# Sliding window parameters
SEQUENCE_LENGTH = 4 # Use the last 4 items as history sequence context features
sequences = []
target_items = []

for user_id, history in user_histories.items():
    if len(history) <= SEQUENCE_LENGTH:
        continue # Skip users who haven't watched enough movies to build a window
        
    # Slide a window across their entire timeline history
    for i in range(len(history) - SEQUENCE_LENGTH):
        # Extract the window context sequence: indices 0, 1, 2, 3
        seq = history[i : i + SEQUENCE_LENGTH]
        # Extract the target next item label: index 4
        target = history[i + SEQUENCE_LENGTH]
        
        sequences.append(seq)
        target_items.append(target)

sequences = np.array(sequences, dtype=np.int64)
target_items = np.array(target_items, dtype=np.int64)

print(f"Engineered {len(sequences)} sequence windows across the catalog.")
print(f"Sample Input Sequence: {sequences[0]} -> Target Next Movie: {target_items[0]}")


# =========================================================================
# 2. PYTORCH CUSTOM SEQUENTIAL DATASET
# =========================================================================
class MovieSequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# =========================================================================
# 3. TRANSFORMER RETRIEVAL MODEL ARCHITECTURE
# =========================================================================
class SequentialTransformerRetrieval(nn.Module):
    def __init__(self, num_movies, embedding_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        # Item Lookup Matrix (64 dimensions)
        self.item_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # Positional Embeddings to learn structural value of time sequence positions (0 to 3)
        self.position_embedding = nn.Embedding(SEQUENCE_LENGTH, embedding_dim)
        
        # Standard Multi-Head Attention Encoder block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True # Keeps shapes formatted as [Batch, Seq, Dim]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final projection head to calculate prediction probability scores over the entire catalog
        self.output_projection = nn.Linear(embedding_dim, num_movies)
        
    def forward(self, sequence_indices):
        batch_size, seq_len = sequence_indices.size()
        
        # 1. Extract Item Lookup Vectors
        word_embeddings = self.item_embedding(sequence_indices) # Shape: [B, 4, 64]
        
        # 2. Build and combine Positional Timestamps
        positions = torch.arange(0, seq_len, device=sequence_indices.device).expand(batch_size, seq_len)
        pos_embeddings = self.position_embedding(positions) # Shape: [B, 4, 64]
        
        # Inject positional values directly into item properties
        x = word_embeddings + pos_embeddings
        
        # 3. Pass through Multi-Head Attention Layers
        transformer_output = self.transformer(x) # Shape: [B, 4, 64]
        
        # 4. Extract representation vector of the final item in the sequence
        # This acts as our mathematically blended sequence profile representation!
        final_session_vector = transformer_output[:, -1, :] # Shape: [B, 64]
        
        # 5. Output raw scores over the whole catalog
        logits = self.output_projection(final_session_vector) # Shape: [B, num_movies]
        return final_session_vector, logits




# =========================================================================
# 4. PRODUCTION TRAINING LOOP EXECUTION
# =========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training session booting up on target hardware accelerator: {device}")

# Initialize Data Utilities
dataset = MovieSequenceDataset(sequences, target_items)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# Instantiate Architecture Framework
model = SequentialTransformerRetrieval(num_movies=num_movies).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Run Optimization for 5 clean training sweeps
EPOCHS = 5
model.train()

for epoch in range(EPOCHS):
    running_loss = 0.0
    for seq_batch, target_batch in dataloader:
        seq_batch, target_batch = seq_batch.to(device), target_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward Pass: Extract sequence representation and class predictions
        _, output_logits = model(seq_batch)
        
        loss = criterion(output_logits, target_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * seq_batch.size(0)
        
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} -> Cross Entropy Sequence Loss: {epoch_loss:.4f}")

# Save the final structural sequential weights safely to disk
MODEL_SAVE_PATH = "./sequential_transformer_retrieval.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\nSuccess! Deep Sequence Transformer weights saved cleanly to: {MODEL_SAVE_PATH}")

