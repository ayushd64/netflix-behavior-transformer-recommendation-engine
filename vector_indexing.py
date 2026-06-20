import numpy as np
import faiss
import time

# =====================================
# 1. STIMULATE A PRODUCTION CATLOG
# =====================================
num_movies = 1000000 # Our catalog size
vector_dimension = 8 # Matching the embedding deimensions from our PyTorch model


# Generate 10,000 random item embeddings (simulating trained neural networks)
np.random.seed(42)
movie_embeddings = np.random.random((num_movies, vector_dimension)).astype('float32')

print(f"Generated a catalog of {num_movies} movies with {vector_dimension}-dimensional embeddings.")

# =====================================
# 2. BRUTE FORCE V/S IVF INDEX SETUP
# =====================================
# Method A: Flat Index (Brute-force exact matching - calculates everything)
index_flat = faiss.IndexFlatL2(vector_dimension)

# Method B: IVF index (Approximate Nearest Neighbors - Partitioned Search)
quantizer = faiss.IndexFlatL2(vector_dimension) # How we measure distance to cluster centers
nlist = 1000 # Number of clusters/neighborhoods to break our space into
index_ivf = faiss.IndexIVFFlat(quantizer, vector_dimension, nlist, faiss.METRIC_L2)

# Train the IVF index so it can learn where to place the 50 cluster centers
index_ivf.train(movie_embeddings)

# Add our movies to both indexes
index_flat.add(movie_embeddings)
index_ivf.add(movie_embeddings)

# =====================================
# 3. PERFORMANCE BENCHMARKING
# =====================================
# Create a fake Target User Embedding (what kind of latent features this user loves)
user_preference = np.random.random((1, vector_dimension)).astype('float32')
top_k = 5

print(f"\nSearching for the top {top_k} movie candidates for our user...\n")

# --- Test 1: Brute Force Search ---
start_time = time.time()
distance_flat, indices_flat = index_flat.search(user_preference, top_k)
flat_time = time.time() - start_time
print(f"\n[Flat Index / Exact Search]   Took {flat_time:.6f} seconds.")
print(f" -> Recommended Movie IDs: {indices_flat[0]}")

# --- Test 2: Accelerated IVF Search ---
# nprobe: how many neighbouring clusters to look inside. Lower = faster, Higher = more accurate.
index_ivf.nprobe = 5
start_time = time.time()
distances_ivf, indices_ivf = index_ivf.search(user_preference, top_k)
ivf_time = time.time() - start_time
print(f"\n[IVF Index / ANN Search]   Took {ivf_time:.6f} seconds.")
print(f" -> Recommended Movie IDs: {indices_ivf[0]}")

# Calculate speedup factor
if ivf_time > 0:
    print(f"\nAn IVF Index search runs {flat_time / ivf_time:.1f}x faster than a linearscan at this scale!")

