from fastapi import FastAPI, HTTPException
import torch
import torch.nn as nn
import json
import os

app = FastAPI(title="Netflix Recommendation Engine API", version="1.0")


# ========================================
# 1. DEFINE ARCHITECTURE (Must match training exactly)
# ========================================
class OptimizedNCF(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=64):
        super(OptimizedNCF, self).__init__()
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
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        return self.output_layer(x).squeeze()



# ========================================
# 2. GLOBAL LIFETIME STATE INITIALIZATION
# ========================================
# Haedcoded to match our processed MovieLens dataset parameters
NUM_USERS = 943
NUM_MOVIES = 1682

# Force to CPU for simple, reliable multi-threaded web endpoints
device = torch.device('cpu')

# Load model structure and inject saved weights
model = OptimizedNCF(NUM_USERS, NUM_MOVIES, embedding_dim=64)
model.load_state_dict(torch.load('./optimized_ncf_model.pth', map_location=device))
model.eval() # Dropouts disabled for live inference

# Load ID index converters
with open('./id_mappings.json', 'r') as f:
    mappings = json.load(f)


user_map = mappings["user_mapping"]
movie_map = mappings["movie_mapping"]
rev_movie_map = mappings["reverse_movie_mapping"]



# ========================================
# 3. DEFINE ENDPOINTS
# ========================================
@app.get("/")
def home():
    return {"status": "Online", "message": "Welcome to the MovieLens Deep Learning Recommmendation API"}

@app.get("/predict")
def predict_rating(user_id: int, movie_id: int):
    # Validate raw IDs against our lookup mappings
    str_user = str(user_id)
    str_movie = str(movie_id)

    if str_user not in user_map or str_movie not in movie_map:
        raise HTTPException(status_code=404, detail="User ID or Movie ID not found in system mappings.")
    
    # Extract structural array codes
    user_idx = user_map[str_user]
    movie_idx = movie_map[str_movie]

    # Convert inputs to torch tensors
    u_tensor = torch.tensor([user_idx], dtype=torch.long).to(device)
    m_tensor = torch.tensor([movie_idx], dtype=torch.long).to(device)

    with torch.no_grad():
        prediction = model(u_tensor, m_tensor)
        predicted_score = round(float(prediction.item()), 2)
    
    return{
        "user_id": user_id,
        "movie_id": movie_id,
        "predicted_rating": min(max(predicted_score, 1.0), 5.0) # Clip predictionsbetween 1 and 5 stars
    }