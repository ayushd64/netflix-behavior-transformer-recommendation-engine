import pandas as pd

# 1. Load the Ratings data
# The ml-100k dataset uses tab_separated values ('\t') and doesn't have a header row
rating_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings_df = pd.read_csv('./data/ml-100k/u.data', sep='\t', names=rating_columns)


# 2. Load the Movie Titles data
# It uses pip separator ('|') and contains specific encoding
movie_columns = ['movie_id', 'title', 'release_data', 'video_release_date', 'IMDB_URL'] + [f'genre_{i}' for i in range(19)]
movies_df = pd.read_csv('./data/ml-100k/u.item', sep='|', names=movie_columns, encoding='latin-1')

# Keep only the movie_id and title for easy viewing
movies_df = movies_df[['movie_id', 'title']]

# 3. Merge them together to see what people are watching\
full_dataset = pd.merge(ratings_df, movies_df, on='movie_id')

print("--- MovieLens 100K Statistics ---")
print(f"Total Ratings Logged: {len(ratings_df)}")
print(f"Unique Users:         {ratings_df['user_id'].nunique()}")
print(f"Unique Movies:        {ratings_df['movie_id'].nunique()}\n")

print("\n--- Most Heavily Reviewed Movies (Most Popular) ---")
print(full_dataset['title'].value_counts().head(5))