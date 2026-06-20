import math

# 1. Our Dataset Matrix
ratings = {
    'Alice': {'Stranger Things': 5, 'Wednesday': 4, 'The Witcher': 1, 'Breaking Bad': 0, 'Narcos': 0},
    'Bob': {'Stranger Things': 0, 'Wednesday': 0, 'The Witcher': 4, 'Breaking Bad': 5, 'Narcos': 2},
    'Charlie': {'Stranger Things': 4, 'Wednesday': 5, 'The Witcher': 0, 'Breaking Bad': 1, 'Narcos': 0},
    'Dwight': {'Stranger Things': 1, 'Wednesday': 0, 'The Witcher': 3, 'Breaking Bad': 4, 'Narcos': 5},
}

# 2. Function to calculate Cosine Similarity between two users
def calculate_cosine_similarity(user1, user2):
    ratings1 = ratings[user1]
    ratings2 = ratings[user2]

    dot_product = 0
    sum_squares1 = 0
    sum_squares2 = 0

    for movie in ratings1:
        rate1 = ratings1[movie]
        rate2 = ratings2[movie]

        dot_product += rate1 * rate2
        sum_squares1 += rate1 ** 2
        sum_squares2 += rate2 ** 2
    
    magnitude1 = math.sqrt(sum_squares1)
    magnitude2 = math.sqrt(sum_squares2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)

# 3. NEW: Function to predict a missing rating
def predict_rating(target_user, target_movie):
    total_similarity_score = 0
    weighted_ratings_sum = 0

    # Loop through every other user in our system
    for other_user in ratings:
        if other_user == target_user:
            continue        # Skip comparing Alice to Alice
        
        # Check if this other user has actually watched the movie
        if ratings[other_user][target_movie] > 0:
            # Step A: Get similarity between target_user and other_user
            sim = calculate_cosine_similarity(target_user, other_user)

            # Step B: Get the rating this other user gave to the target movie
            rating = ratings[other_user][target_movie]

            # Step C: Accumulate values for the weighted average
            weighted_ratings_sum += sim * rating
            total_similarity_score += sim
    
    # If no one else watched the movie, we can't predict a score
    if total_similarity_score == 0:
        return 0.0
    
    # Step D: Final calculation
    return weighted_ratings_sum / total_similarity_score

# 3. Test it out
similarity_ac = calculate_cosine_similarity('Alice', 'Charlie')
similarity_ad = calculate_cosine_similarity('Alice', 'Dwight')

print("--- Netflix Recommendation System: Phase 1 ---")
print(f"Similarity between Alice and Charlie: {similarity_ac:.3f}")
print(f"Similarity between Alice and Dwight:  {similarity_ad:.3f}")

# --- Test the Prediction Engine ---
print("--- Netflix Recommendation System: Phase 1 (Predictions) ---")
predicted_score = predict_rating('Alice', 'Breaking Bad')
print(f"Predicted rating for Alice on 'Breaking Bad': {predicted_score:.2f} stars")