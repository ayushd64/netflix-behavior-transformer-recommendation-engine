from fastapi import FastAPI, HTTPException, Query
import torch
import torch.nn as nn
import faiss
import json
import os
import numpy as np

app = FastAPI(title="Netflix Ultra Production Transformer-Backed API", version="5.0")

# ==========================================
# 1. CORE ARCHITECTURE LAYER DEFINITIONS
# ==========================================

# STAGE 1 RETRIEVAL BACKBONE: The Behavior Sequence Transformer Encoder
class SequentialTransformerRetrieval(nn.Module):
    def __init__(self, num_movies=1682, embedding_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.item_embedding = nn.Embedding(num_movies, embedding_dim)
        self.position_embedding = nn.Embedding(4, embedding_dim) # Expects max 5 items now
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(embedding_dim, num_movies)
        
    def forward(self, sequence_indices):
        batch_size, seq_len = sequence_indices.size()
        word_embeddings = self.item_embedding(sequence_indices)
        positions = torch.arange(0, seq_len, device=sequence_indices.device).expand(batch_size, seq_len)
        pos_embeddings = self.position_embedding(positions)
        x = word_embeddings + pos_embeddings
        transformer_output = self.transformer(x)
        # Extract the sequence profile vector representing their aggregated intent
        return transformer_output[:, -1, :] 

# STAGE 2 RANKING BACKBONE: Deep Neural Collaborative Filtering Neural Net
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
        
    def forward(self, user_indices, item_indices):
        user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding(item_indices)
        x = torch.cat([user_vec, item_vec], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.output_layer(x).squeeze()

# ==========================================
# 2. STATE INFRASTRUCTURE COMPILATION
# ==========================================
INDEX_PATH = "./realtime_movies.index"
MAPPINGS_PATH = "./realtime_mappings.json"
RANKING_MODEL_PATH = "./optimized_ncf_model.pth"
TRANSFORMER_MODEL_PATH = "./sequential_transformer_retrieval.pth"

device = torch.device("cpu") # Keep inference flat on CPU for server thread reliability

# Load raw indexes and maps
index = faiss.read_index(INDEX_PATH)
with open(MAPPINGS_PATH, "r") as f:
    mappings = json.load(f)
movie_idx_to_title = mappings["movie_idx_to_title"]
movie_title_to_idx = mappings["movie_title_to_idx"]

# Instantiate and freeze Stage 1 Transformer weights
transformer_model = SequentialTransformerRetrieval()
transformer_model.load_state_dict(torch.load(TRANSFORMER_MODEL_PATH, map_location=device), strict=False)
transformer_model.eval()

# Instantiate and freeze Stage 2 Neural Ranker weights
ranking_model = OptimizedNCF()
ranking_model.load_state_dict(torch.load(RANKING_MODEL_PATH, map_location=device))
ranking_model.eval()

USER_SESSIONS = {}

# ==========================================
# 3. COMPONENT INTERACTION CONTROLLER
# ==========================================
@app.get("/")
def home():
    return {"status": "Online", "mode": "Transformer-Attention Two-Stage Pipeline Activated"}

@app.get("/watch/")
def watch_and_recommend(
    user_id: int = Query(..., description="Target database user index"),
    title: str = Query(..., description="Name of movie just clicked")
):
    target_title = title.strip()
    if target_title not in movie_title_to_idx:
        raise HTTPException(status_code=404, detail="Movie not found in catalog.")
    
    # Session Append Updates
    str_user = str(user_id)
    if str_user not in USER_SESSIONS:
        USER_SESSIONS[str_user] = []
    
    USER_SESSIONS[str_user].append(target_title)
    USER_SESSIONS[str_user] = USER_SESSIONS[str_user][-4:] # Max length of 5 
    current_history = USER_SESSIONS[str_user]
    
    # Convert text strings history directly to internal integer index codes
    sequence_indices = [int(movie_title_to_idx[t]) for t in current_history]
    
    # Padding step if user session is brand new (pre-pad with the first movie ID)
    while len(sequence_indices) < 4:
        sequence_indices.insert(0, sequence_indices[0])
        
    # -------------------------------------------------------------------------
    # STAGE 1: INFERENCE VIA MULTI-HEAD SELF ATTENTION 
    # -------------------------------------------------------------------------
    seq_tensor = torch.tensor([sequence_indices], dtype=torch.long)
    
    with torch.no_grad():
        # The Transformer processes the click flow sequence to compute a search vector
        context_session_vector = transformer_model(seq_tensor).numpy().astype('float32')
        
    # Search our Faiss vector database space for 20 deep candidate matches
    candidate_pool_size = 20 + len(current_history)
    _, candidate_indices = index.search(context_session_vector, candidate_pool_size)
    
    retrieved_candidates = []
    for idx in candidate_indices[0]:
        str_idx = str(idx)
        if str_idx in movie_idx_to_title:
            cand_title = movie_idx_to_title[str_idx]
            if cand_title not in current_history:
                retrieved_candidates.append(idx)
                
    retrieved_candidates = retrieved_candidates[:20]
    
    # -------------------------------------------------------------------------
    # STAGE 2: PRECISE NEURAL RANKING SCORING
    # -------------------------------------------------------------------------
    internal_user_idx = user_id if user_id < 943 else 0
    user_tensor = torch.tensor([internal_user_idx] * len(retrieved_candidates), dtype=torch.long)
    movie_tensor = torch.tensor(retrieved_candidates, dtype=torch.long)
    
    with torch.no_grad():
        predicted_ratings = ranking_model(user_tensor, movie_tensor)
        if len(retrieved_candidates) == 1:
            scores = [float(predicted_ratings.item())]
        else:
            scores = predicted_ratings.numpy().tolist()
            
    scored_catalog = []
    for position, movie_idx in enumerate(retrieved_candidates):
        title_string = movie_idx_to_title[str(movie_idx)]
        score_value = round(scores[position], 2)
        scored_catalog.append({"title": title_string, "predicted_rating": score_value})
        
    final_ranked_recommendations = sorted(scored_catalog, key=lambda x: x["predicted_rating"], reverse=True)
    
    return {
        "user_id": user_id,
        "transformer_tracked_sequence": current_history,
        "pipeline_strategy": "Stage 1: Multi-Head Transformer Attention -> Stage 2: Deep NCF Scoring",
        "top_5_recommendations": final_ranked_recommendations[:5]
    }

