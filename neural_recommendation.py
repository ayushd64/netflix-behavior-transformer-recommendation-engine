import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# SETUP THE DEVICE (CPU V/S GPU)
# This automatically picks the GPU if available, otherwise defaults to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 1. Prepare Training Data from our Matrix
# Rows: Alice(0), Bob(1), Charlie(2), Dwight(3)
# Cols: Stranger Things(0), Wednesday(1), The Witcher(2), Breaking Bad(3), Narcos(4)
R = np.array([
    [5, 4, 1, 0, 0],  # Alice
    [0, 0, 4, 5, 2],  # Bob
    [4, 5, 0, 1, 0],  # Charlie
    [1, 0, 3, 4, 5]   # Dwight
])

# Convert our known ratings into lists of [User_ID, Movie_ID] and [Rating] for PyTorch
user_indices, item_indices, ratings = [], [], []
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        if R[i, j] > 0: # Only consider known ratings
            user_indices.append(i)
            item_indices.append(j)
            ratings.append(R[i, j])

# Convert lists to PyTorch tensors
users_t = torch.tensor(user_indices, dtype=torch.long).to(device)
items_t = torch.tensor(item_indices, dtype=torch.long).to(device)
ratings_t = torch.tensor(ratings, dtype=torch.float32).to(device)

# 2. Define the Neural Network Architecture
class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=0):
        super(NeuralCollaborativeFiltering, self).__init__()

        # Embedding Layers (Learnable Latent Factor Matrices)
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim) 

        # Deep Neural Network Layers
        # Input size is embedding_dim * 2 because we glue user and item embeddings together
        self.fc1 = nn.Linear(embedding_dim * 2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.output_layer = nn.Linear(8, 1)

        self.relu = nn.ReLU()

    def forward(self, user_indices, item_indices):
        # Step A: Look up the dense vector representations
        user_vector = self.user_embedding(user_indices)
        item_vector = self.item_embedding(item_indices)

        # Step B: Concatenate the user and item vectors together
        x = torch.cat([user_vector, item_vector], dim=-1)

        # Step C: Pass through the hidden deep layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        # Step D: Predict the final rating
        prediction = self.output_layer(x)
        return prediction.squeeze()
    

# 3. Instantiate the Model, Loss Function, and Optimizer
model = NeuralCollaborativeFiltering(num_users=4, num_items=5, embedding_dim=4).to(device)
criterion = nn.MSELoss() # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.02)

print("Training Deep Learning Recommendation Model via PyTorch...\n")

# 4. The Deep Learning Training Loop
epochs = 500
for epoch in range(epochs):
    model.train()

    # Forward Pass: Predict ratings for all known coordinates
    predictions = model(users_t, items_t)

    # Calculate how wrong the network's current weights are
    loss = criterion(predictions, ratings_t)

    # Backward Pass: Calculate gradients and optiize weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Training Loss (MSE): {loss.item():.4f}")

# 5. Predict Alice's rating for Breaking Bad (User 0, Movie 3)
model.eval()
with torch.no_grad():
    alice_tensor = torch.tensor([0], dtype=torch.long).to(device)
    breaking_bad_tensor = torch.tensor([3], dtype=torch.long).to(device)

    predicted_rating = model(alice_tensor, breaking_bad_tensor)
    print(f"\n[Deep Learning Prediction] Alice's predicted rating for 'Breaking Bad': {predicted_rating.item():.2f} stars")
