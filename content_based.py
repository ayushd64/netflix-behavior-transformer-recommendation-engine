from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Our Content Dataset (Metadata)
movies = [ 
    {"title": "Stranger Things", "description": "A group of young friends witness supernatural forces and secret government exploits in a small town."}, 
    {"title": "Wednesday", "description": "A sleuthing, supernaturally infused mystery charting Wednesday Addams' years as a student at Nevermore Academy."}, 
    {"title": "The Witcher", "description": "Geralt of Rivia, a mutated monster-hunter for hire, journeys toward his destiny in a turbulent world of magic and beasts."}, 
    {"title": "Breaking Bad", "description": "A high school chemistry teacher diagnosed with inoperable lung cancer turns to manufacturing and selling methamphetamine to secure his family's future."}, 
    {"title": "Narcos", "description": "A chronicled look at the criminal exploits of Colombian drug lord Pablo Escobar and the gritty drug cartels."}, 
    {"title": "Interstellar", "description": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival amid space travel."} # Brand new movie! 
] 

# Extract descriptions and titles
descriptions = [movie["description"] for movie in movies]
titles = [movie["title"] for movie in movies]

# 2. Compute TF-IDF Matrix
# This converts sentences intoa matrix of numerical imporatance scores
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(descriptions)

# 3. Calculate Content Similarity Matrix using Cosine Similarity
# It computes the text vectors of every movie against every other movie
content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 4. Function to get recommendations based on content
def get_content_recommendations(target_movie_title, top_n=2):
    # Find the index of the movie that matches the title
    idx = titles.index(target_movie_title)

    # Get the pairwise similarity scores of all movies with this movie
    sim_scores = list(enumerate(content_sim[idx]))

    # Sort the movies based on similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    print(f"\nMovies most similar to '{target_movie_title}':")
    count = 0
    for i, score in sim_scores:
        if titles[i] != target_movie_title: # Skip the same movie
            print(f" -> {titles[i]} (Similarity Score: {score:.3f})")
            count += 1
            if count == top_n:
                break

# --- Test our Content Engine ---
print("--- Netflix Recommendation System: Phase 2 (Content-Based Filtering) ---")
get_content_recommendations("Breaking Bad", top_n=2)
get_content_recommendations("Stranger Things", top_n=2)
