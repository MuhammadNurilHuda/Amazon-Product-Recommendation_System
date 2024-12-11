# Import libraries
import pandas as pd
import numpy as np
import os, re
from tqdm import tqdm
from typing import List
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import FastText
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity

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

# Clean currency columns
def clean_currency(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Cleans currency columns by removing symbols and converting them to float.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (List[str]): List of column names to clean.

    Returns:
        pd.DataFrame: DataFrame with cleaned currency columns.
    """
    for col in tqdm(columns, desc="Cleaning Currency Columns"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('₹', '', regex=False)
            df[col] = df[col].str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")
    return df

clean_currency(combined_df, ['discount_price', 'actual_price'])

# Clean numeric columns
def clean_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Cleans numeric columns by removing commas and converting them to numeric types.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (List[str]): List of column names to clean.

    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns.
    """
    for col in tqdm(columns, desc="Cleaning Numeric Columns"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")
    return df

clean_columns(combined_df, ['ratings', 'no_of_ratings'])

# Handle duplicates and missing values
combined_df.drop_duplicates(subset=['name', 'main_category', 'actual_price'], inplace=True)
combined_df['actual_price'] = combined_df.groupby('sub_category')['actual_price'].transform(lambda x: x.fillna(x.median()))
combined_df.fillna({'ratings': 0, 'no_of_ratings': 0, 'discount_price': 0}, inplace=True)

# Content-based filtering preprocessing
def preprocess_content_based(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the DataFrame for content-based filtering by creating combined features,
    removing stopwords, and applying lemmatization.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for content-based filtering.
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_tokens(tokens):
        tokens = [word.lower() for word in tokens]
        tokens = [re.sub(r'[^\w\s]', '', word) for word in tokens if word]
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return tokens

    tqdm.pandas(desc="Processing Content-Based Tokens")
    df['combined_features'] = df['main_category'] + ' ' + df['sub_category'] + ' ' + df['name']
    df['tokenized_features'] = df['combined_features'].progress_apply(lambda x: x.split())
    df['cleaned_features'] = df['tokenized_features'].progress_apply(preprocess_tokens)
    return df

df_content_based = preprocess_content_based(combined_df)

# Load sparse similarity matrix
try:
    sparse_similarity_matrix = load_npz("models/sparse_similarity_matrix.npz")
    print("Sparse similarity matrix loaded successfully.")
except Exception as e:
    print(f"Failed to load sparse similarity matrix: {e}")
    # If loading fails, compute the similarity matrix
    print("Computing sparse similarity matrix from scratch...")
    embedding_matrix = np.stack(df_content_based['fasttext_embedding'].values)
    sparse_similarity_matrix = compute_sparse_similarity(embedding_matrix, threshold=0.7)
    save_npz("models/sparse_similarity_matrix.npz", sparse_similarity_matrix)
    print("Sparse similarity matrix saved successfully.")

# Recommendation function
def recommend_products(product_index: int, similarity_matrix: csr_matrix, df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Generates product recommendations based on similarity scores.

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

# Recommendation pipeline
def recommendation_pipeline(product_index: int, similarity_matrix: csr_matrix, df: pd.DataFrame, top_n: int = 5):
    """
    Generates recommendations and displays them.

    Args:
        product_index (int): Index of the selected product.
        similarity_matrix (csr_matrix): Sparse similarity matrix.
        df (pd.DataFrame): DataFrame containing product metadata.
        top_n (int): Number of recommendations. Default is 5.

    Returns:
        None
    """
    print("Selected Product:")
    print(df.iloc[product_index][['name', 'main_category', 'sub_category']])
    recommendations = recommend_products(product_index, similarity_matrix, df, top_n)
    print("\nRecommendations:")
    print(recommendations)

# Example usage
product_index = 250
recommendation_pipeline(product_index, sparse_similarity_matrix, df_content_based, top_n=5)
