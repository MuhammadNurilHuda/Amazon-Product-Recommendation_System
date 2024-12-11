# Import Libraries
import pandas as pd
import numpy as np
import os, re
from tqdm import tqdm
from typing import List
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, save_npz, load_npz

# Set pandas options
pd.set_option('display.max_colwidth', None)

# Load dataset
def load_datasets(folder_path: str, file_extension: str = '.csv') -> pd.DataFrame:
    """
    Loads and combines all CSV files from the specified folder into a single DataFrame.

    Args:
        folder_path (str): Path to the folder containing CSV files.
        file_extension (str): File extension to filter files. Default is '.csv'.

    Returns:
        pd.DataFrame: Combined DataFrame from all CSV files in the folder.
    """
    files = [file for file in os.listdir(folder_path) if file.endswith(file_extension)]
    dataframes = []

    for file in tqdm(files, desc="Loading CSV Files"):
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(file_path)
            if 'Unnamed: 0' in df.columns:
                df.drop(columns=['Unnamed: 0'], inplace=True)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading file {file}: {e}")

    return pd.concat(dataframes, ignore_index=True)

# Load datasets
combined_df = load_datasets('Dataset')
combined_df.drop(columns=['image', 'link'], inplace=True)

# Ensure 'ratings' column is numeric
combined_df['ratings'] = pd.to_numeric(combined_df['ratings'], errors='coerce')
combined_df['ratings'] = combined_df['ratings'].fillna(0)

# Clean product names
def clean_product_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the 'name' column in the DataFrame to ensure consistency.

    Cleaning steps:
    1. Remove stopwords.
    2. Convert text to lowercase.
    3. Remove punctuations.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'name' column.

    Returns:
        pd.DataFrame: DataFrame with cleaned 'name' column.
    """
    stop_words = set(stopwords.words('english'))
    
    def clean_text(text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            text = ' '.join([word for word in text.split() if word not in stop_words])
        return text

    tqdm.pandas(desc="Cleaning Product Names")
    df['name'] = df['name'].progress_apply(clean_text)
    return df

# Clean product names
combined_df = clean_product_names(combined_df)

# Reduce dataset to 10,000 rows randomly
print("Reducing dataset to 10,000 rows for performance...")
reduced_df_collaborative = combined_df.sample(n=10000, random_state=42).reset_index(drop=True)

# Add simulated users for collaborative filtering
reduced_df_collaborative['user'] = np.random.randint(1, 1000, size=len(reduced_df_collaborative))

# Create pivot table for user-item matrix
print("Creating user-item pivot table...")
pivot_table = reduced_df_collaborative.pivot_table(index='user', columns='name', values='ratings', fill_value=0)

# Transpose pivot table for item-based similarity calculation
print("Transposing pivot table for similarity calculation...")
product_matrix = csr_matrix(pivot_table.T)

# Compute sparse cosine similarity
def compute_sparse_similarity(matrix: csr_matrix, threshold: float = 0.7) -> csr_matrix:
    """
    Computes a sparse cosine similarity matrix.

    Args:
        matrix (csr_matrix): Sparse matrix to compute similarities for.
        threshold (float): Minimum similarity threshold to include in the sparse matrix.

    Returns:
        csr_matrix: Sparse cosine similarity matrix.
    """
    n = matrix.shape[0]
    data, rows, cols = [], [], []

    for i in tqdm(range(n), desc="Computing Sparse Similarity"):
        sim_scores = cosine_similarity(matrix[i], matrix).flatten()
        for j, score in enumerate(sim_scores):
            if score >= threshold and i != j:  # Exclude self-similarity
                data.append(score)
                rows.append(i)
                cols.append(j)

    return csr_matrix((data, (rows, cols)), shape=(n, n))

# Calculate sparse similarity matrix
try:
    print("Attempting to load existing sparse similarity matrix...")
    sparse_similarity_matrix = load_npz("models/sparse_similarity_matrix_collab.npz")
    print("Sparse similarity matrix loaded successfully.")
except Exception as e:
    print(f"Failed to load sparse similarity matrix: {e}")
    print("Computing sparse similarity matrix from scratch...")
    sparse_similarity_matrix = compute_sparse_similarity(product_matrix, threshold=0.7)
    save_npz("models/sparse_similarity_matrix_collab.npz", sparse_similarity_matrix)
    print("Sparse similarity matrix saved successfully.")

# Recommendation function
def recommend_products_sparse(product_index: int, similarity_matrix: csr_matrix, df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Generates product recommendations based on sparse similarity scores.

    Args:
        product_index (int): Index of the product to base recommendations on.
        similarity_matrix (csr_matrix): Sparse similarity matrix.
        df (pd.DataFrame): DataFrame containing product metadata.
        top_n (int): Number of recommendations. Default is 5.

    Returns:
        pd.DataFrame: DataFrame of recommended products.
    """
    sim_scores = similarity_matrix[product_index].toarray().flatten()
    top_indices = sim_scores.argsort()[::-1][1:top_n+1]
    recommendations = df.iloc[top_indices][['name', 'main_category', 'sub_category', 'ratings']].copy()
    recommendations['similarity_score'] = sim_scores[top_indices]
    return recommendations

# Select a product for recommendations
product_index = 250
recommendations = recommend_products_sparse(product_index, sparse_similarity_matrix, reduced_df_collaborative, top_n=5)

if recommendations is not None:
    print("Selected Product:")
    print(reduced_df_collaborative.iloc[product_index][['name', 'main_category', 'sub_category']])
    print("\nRecommendations:")
    print(recommendations)
