import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===========================================
# DATA & METADATA SETUP
# ===========================================
titles = ["Stranger Things", "Wednesday", "The Witcher", "Breaking Bad", "Narcos"]
descriptions = [
    "A group of young friends witness supernatural forces and secret government exploits in a small town.",
    "A sleuthing, supernaturally infused mystery charting Wednesday Addams' years as a student at Nevermore Academy.",
    "Geralt of Rivia, a mutated monster-hunter for hire, journeys toward his destiny in a turbulent world of magic and beasts.",
    "A high school chemistry teacher diagnosed with inoperable lung cancer turns to manufacturing and selling methamphetamine to secure his family's future.",
    "A chronicled look at the criminal exploits of Colombian drug lord Pablo Escobar and the gritty drug cartels."
]

# Original incomplete User-Item rating matrix
R = np.array([
    [5, 4, 1, 0, 0],  # Alice
    [0, 0, 4, 5, 2],  # Bob
    [4, 5, 0, 1, 0],  # Charlie
    [1, 0, 3, 4, 5]   # Dwight
])

user_mapping = {"Alice": 0, "Bob": 1, "Charlie": 2, "Dwight": 3}

# ===========================================
# STEP 1: COMPUTE CONTENT SIMILARITY MATRIX
# ===========================================
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(descriptions)
content_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)


# ===========================================
# STEP 2: TRAIN MATRIX FACTORIZATION (COLLABORATIVE)
# ===========================================
num_users, num_items = R.shape
K = 2 # Number of Latent Factors
np.random.seed(42)
P = np.random.normal(scale=1.0, size=(num_users, K))
Q = np.random.normal(scale=1.0, size=(num_items, K))
b_u = np.zeros(num_users) # User Biases
b_i = np.zeros(num_items) # Item Biases
global_mean = np.mean(R[R > 0])

alpha = 0.05 # Learning Rate
beta = 0.02 # Regularization penalty
epochs = 1000 # Reduced epochs for faster convergence in this example

for epoch in range(epochs):
    for i in range(num_users):
        for j in range(num_items):
            if R[i, j] > 0:
                prediction = global_mean + b_u[i] + b_i[j] + np.dot(P[i, :], Q[j, :])
                error = R[i, j] - prediction

                b_u[i] += alpha * (error - beta * b_u[i])
                b_i[j] += alpha * (error - beta * b_i[j])
                P[i, :] += alpha * (error * Q[j, :] - beta * P[i, :])
                Q[j, :] += alpha * (error * P[i, :] - beta * Q[j, :])


# Build full collaborative predictions matrix
collab_predictions = np.zeros((num_users, num_items))
for i in range(num_users):
    for j in range(num_items):
        pred = global_mean + b_u[i] + b_i[j] + np.dot(P[i, :], Q[j, :])
        collab_predictions[i, j] = np.clip(pred, 1.0, 5.0)


# ===========================================
# STEP 3: HYBRID RECOMMENDATION FUNCTION
# ===========================================
def get_hybrid_recommendations(user_name, top_n=2):
    user_idx = user_mapping[user_name]
    print(f"\n--- Generating Hybrid Recommendations for {user_name} ---")

    hybrid_scores = []

    for item_idx, movie_title in enumerate(titles):
        # We only want to recommend movies the user hasn't watched yet
        if R[user_idx, item_idx] == 0:
            # A. Get Collaborative Filter prediction score
            collab_score = collab_predictions[user_idx, item_idx]

            # B. Get Content Filter Score (average similarity to items user rated highly >= 4)
            high_rated_indices = [idx for idx, rating in enumerate(R[user_idx]) if rating >= 4]
            if high_rated_indices:
                content_score = np.mean([content_sim_matrix[item_idx, r_idx] for r_idx in high_rated_indices])
            else:
                content_score = 0
            
            # C. Fuse the scores with weights (80% collaborative, 20% content scaled to 5 stars)
            final_score = (0.8 * collab_score) + (0.2 * (content_score * 5)) # Scale content score to 5 stars
            hybrid_scores.append((movie_title, final_score, collab_score, content_score))

    # Sort by final hybrid score in descending order
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)

    for title, score, c_score, txt_score in hybrid_scores[:top_n]:
        print(f"Movie: {title:<16} | Hybrid Rank Score: {score:.2f} (Collab Match: {c_score:.2f}, Text Match: {txt_score:.2f})")


# Run recommendations for our edge-case users
get_hybrid_recommendations("Alice") # Alice has only rated 3 movies, so content similarity will help!
get_hybrid_recommendations("Bob")   # Bob has only rated 3 movies, so content similarity will help!