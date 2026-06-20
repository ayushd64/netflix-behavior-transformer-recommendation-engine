import faiss
import json
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

# 1. LOAD INDEX AND MAPPINGS
index = faiss.read_index("./realtime_movies.index")
with open("./realtime_mappings.json", "r") as f:
    mappings = json.load(f)
movie_idx_to_title = mappings["movie_idx_to_title"]

# 2. EXTRACT VECTORS FROM FAISS
num_movies = index.ntotal
dimensions = index.d
# Reconstruct all vectors back into a NumPy array
vectors = np.vstack([index.reconstruct(i) for i in range(num_movies)])

# 3. COMPRESS 64-DIMENSIONS DOWN TO 2-DIMENSIONS USING PCA
# Principal Component Analysis finds the axes of maximum variance
pca = PCA(n_components=2, random_state=42)
vectors_2d = pca.fit_transform(vectors)

# 4. BUILD A CLEAN DATAFRAME FOR PLOTLY
titles = [movie_idx_to_title.get(str(i), f"Unknown {i}") for i in range(num_movies)]

plot_df = pd.DataFrame({
    'X': vectors_2d[:, 0],
    'Y': vectors_2d[:, 1],
    'Movie Title': titles
})

# 5. GENERATE INTERACTIVE SCATTER PLOT
fig = px.scatter(
    plot_df, 
    x='X', 
    y='Y', 
    hover_name='Movie Title',
    title='Faiss Vector Space Visualization (PCA 2D Projection)',
    template='plotly_dark' # Matches VS Code dark mode aesthetics
)

# Adjust marker layouts for readability
fig.update_traces(marker=dict(size=6, opacity=0.7, line=dict(width=0.5, color='DarkSlateGrey')))
fig.show()

