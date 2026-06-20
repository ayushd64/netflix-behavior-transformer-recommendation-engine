import urllib.request
import zipfile
import os

# Create a data directory if it doesn't exist

os.makedirs('./data', exist_ok=True)

# URL for the MovieLens 100k dataset
url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
zip_path = "./data/ml-100k.zip"

print("Downloading MovieLens 100k dataset... This might take a moment.")
urllib.request.urlretrieve(url, zip_path)

print("Extracting files...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('./data')

print("Dataset successfully downloaded and extracted into the './data/ml-100k' folder!")