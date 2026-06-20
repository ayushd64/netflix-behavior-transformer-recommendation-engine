import numpy as np

# 1. Define our User-Item Matrix using NumPy (0 means unrated)
# Rows: Alice, Bob, Charlie, Dwight
# Columns: Stranger Things, Wednesday, The Witcher, Breaking Bad, Narcos
R = np.array([
    [5, 4, 1, 0, 0],  # Alice
    [0, 0, 4, 5, 2],  # Bob
    [4, 5, 0, 1, 0],  # Charlie
    [1, 0, 3, 4, 5]   # Dwight
])

num_users, num_items = R.shape
K = 2 # Number of Latent Factors (e.g., Sci-Fi preference, Drama Preference)

# 2. Initialize User Matrix (P) and Item Matrix (Q) with random numbers
# P dimensions: (num_users x K)
# Q dimensions: (num_items x K) -> We transpose it to K x num_items for easy math
np.random.seed(42) # For reproucible results
P = np.random.normal(scale=1.0, size=(num_users, K))
Q = np.random.normal(scale=1.0, size=(num_items, K))

b_u = np.zeros(num_users) # User Biases
b_i = np.zeros(num_items) # Item Biases
# Calculate global average of only the watched movies (ratings > 0)
global_mean = np.mean(R[R > 0])

# 3. Hyperparameters for training
alpha = 0.05 # Learning Rate
beta = 0.02 # Regularization penalty to prevent overfitting (the leash!)
epochs = 2000 # How many times to loop over the dataset

print("Starting Matrix Factorization via Stochastic Gradient Descent with Regularization & Biases...\n")

# 4. The Training Loop (SGD)
for epoch in range(epochs):
    for i in range(num_users):
        for j in range(num_items):
            if R[i, j] > 0: # Only train on actual ratings (ignore the 0s)
                # Calculate current predicition: Dot Product of User i and Item j(prediction formula with Biases)
                prediction = global_mean + b_u[i] + b_i[j] + np.dot(P[i, :], Q[j, :])

                # Calculate the Error (Real rating - Predicted rating)
                error = R[i, j] - prediction

                # Update Biases with regularization
                b_u[i] += alpha * (error - beta * b_u[i])
                b_i[j] += alpha * (error - beta * b_i[j])

                # update User and Item Matrices using the Gradient rule(with regularization penalties)
                P[i, :] += alpha * (error * Q[j, :] - beta * P[i, :])
                Q[j, :] += alpha * (error * P[i, :] - beta * Q[j, :])

# 5. Generate the Final Predicted Full Matrix
# Multiplying P and Q^T gives us back the original matrix shape, but completely filled!
Full_Predictions = np.zeros((num_users, num_items))
for i in range(num_users):
    for j in range(num_items):
        pred = global_mean + b_u[i] + b_i[j] + np.dot(P[i, :], Q[j, :])
        # Clip values to keep them strictly between 1.0 and 5.0 stars
        Full_Predictions[i, j] = np.clip(pred, 1.0, 5.0)

print("Original Matrix (with missing values as 0):")
print(R)
print("\nFully Factored Completed Matrix (Predicitions):")
print(np.round(Full_Predictions, 2))