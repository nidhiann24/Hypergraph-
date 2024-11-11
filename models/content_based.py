import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
import math
from typing import List, Tuple
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.data_loader import MovieDataLoader

class ContentBasedRecommender:
    """"
    ContentBasedRecommender is a class that implements a content-based recommendation system for movies.
    Methods:
        __init__():
            Initializes the recommender with an empty dataset.
        fit(df: pd.DataFrame):
            Fits the recommender with the provided dataset.
        gaussian_filter(x: float, y: float, sigma: float) -> float:
            Applies a Gaussian filter to calculate similarity between two values.
        get_entry_variables(id_entry: int) -> List[str]:
            Retrieves all relevant variables (e.g., director, actors, keywords) for a given movie entry.
        add_variables(variables: List[str]) -> pd.DataFrame:
            Creates binary columns for each variable in the dataset and fills them based on the presence of the variables.
        is_sequel(title_1: str, title_2: str, threshold: int = 50) -> bool:
            Checks if two movies are sequels based on title similarity using fuzzy matching.
        get_recommendations(movie_id: int, n_recommendations: int = 5) -> List[Tuple[str, int]]:
            Generates movie recommendations based on content similarity for a given movie ID.
        evaluate_model(test_set: List[Tuple[int, List[int]]], k: int = 5) -> dict:
            Evaluates the recommender system using multiple metrics (precision@k, recall@k, f1@k, ndcg@k, rmse) on a test set.
        """
    def __init__(self):
        self.df = None
        
    def fit(self, df: pd.DataFrame):
        """Fit the recommender with the dataset"""
        self.df = df.copy(deep=True)
        
    def gaussian_filter(self, x: float, y: float, sigma: float) -> float:
        """Apply Gaussian filter to calculate similarity"""
        return math.exp(-(x - y) ** 2 / (2 * sigma ** 2))
        
    def get_entry_variables(self, id_entry: int) -> List[str]:
        """Get all relevant variables for a given movie entry"""
        col_labels = []
        
        # Add director
        if pd.notnull(self.df['director_name'].iloc[id_entry]):
            col_labels.extend(self.df['director_name'].iloc[id_entry].split('|'))
            
        # Add actors
        for i in range(3):
            column = f'actor_{i + 1}_name'
            if pd.notnull(self.df[column].iloc[id_entry]):
                col_labels.extend(self.df[column].iloc[id_entry].split('|'))
                
        # Add keywords
        if pd.notnull(self.df['plot_keywords'].iloc[id_entry]):
            col_labels.extend(self.df['plot_keywords'].iloc[id_entry].split('|'))
            
        return col_labels
        
    def add_variables(self, variables: List[str]) -> pd.DataFrame:
        """Create binary columns for each variable"""
        df_copy = self.df.copy(deep=True)
        
        # Initialize new columns
        for s in variables:
            df_copy[s] = pd.Series([0 for _ in range(len(df_copy))])
            
        # Fill binary values
        columns = ['genres', 'actor_1_name', 'actor_2_name',
                   'actor_3_name', 'director_name', 'plot_keywords']
                   
        for category in columns:
            for index, row in df_copy.iterrows():
                if pd.isnull(row[category]):
                    continue
                for s in row[category].split('|'):
                    if s in variables:
                        df_copy.at[index, s] = 1
                        
        return df_copy
        
    def is_sequel(self, title_1: str, title_2: str, threshold: int = 50) -> bool:
        """Check if two movies are sequels based on title similarity"""
        return (fuzz.ratio(title_1, title_2) > threshold or 
                fuzz.token_set_ratio(title_1, title_2) > threshold)
        
    def get_recommendations(self, movie_id: int, n_recommendations: int = 5) -> List[Tuple[str, int]]:
        """Get movie recommendations based on content similarity"""
        # Get variables for the target movie
        variables = self.get_entry_variables(movie_id)
        
        # Add genre variables
        genre_set = set()
        for s in self.df['genres'].str.split('|').values:
            if isinstance(s, (list, np.ndarray)):
                genre_set = genre_set.union(set(s))
        variables.extend(list(genre_set))
        
        # Create feature matrix
        df_features = self.add_variables(variables)
        X = df_features[variables].values
        
        # Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=n_recommendations + 1, metric='euclidean')
        nbrs.fit(X)
        
        # Get recommendations
        _, indices = nbrs.kneighbors(X[movie_id].reshape(1, -1))
        
        # Format results
        recommendations = []
        for idx in indices[0][1:]:  # Skip the first one as it's the input movie
            recommendations.append(
                (self.df.iloc[idx]['movie_title'], idx)
            )
            
        return recommendations

    def evaluate_model(self, test_set: List[Tuple[int, List[int]]], k: int = 5) -> dict:
        """
        Evaluate the recommender system using multiple metrics
        
        Args:
            test_set: List of tuples (movie_id, relevant_movie_ids)
            k: Number of recommendations to generate
        
        Returns:
            dict: Dictionary containing different evaluation metrics
        """
        metrics = {
            'precision@k': [], 
            'recall@k': [], 
            'f1@k': [], 
            'ndcg@k': [],
            'rmse': []
        }
        
        for movie_id, relevant_ids in test_set:
            # Get recommendations
            recommendations = self.get_recommendations(movie_id, k)
            recommended_ids = [rec[1] for rec in recommendations]
            
            # Calculate Precision@K
            hits = len(set(recommended_ids) & set(relevant_ids))
            precision = hits / k if k > 0 else 0
            metrics['precision@k'].append(precision)
            
            # Calculate Recall@K
            recall = hits / len(relevant_ids) if len(relevant_ids) > 0 else 0
            metrics['recall@k'].append(recall)
            
            # Calculate F1-Score
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            metrics['f1@k'].append(f1)
            
            # Calculate NDCG@K
            dcg = 0
            idcg = 0
            for i, rec_id in enumerate(recommended_ids):
                rel = 1 if rec_id in relevant_ids else 0
                dcg += rel / np.log2(i + 2)
            
            # Calculate IDCG
            for i in range(min(k, len(relevant_ids))):
                idcg += 1 / np.log2(i + 2)
                
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics['ndcg@k'].append(ndcg)
            
            # Calculate RMSE (using binary relevance)
            squared_errors = []
            for rec_id in recommended_ids:
                actual = 1 if rec_id in relevant_ids else 0
                predicted = 1  # Since these are top-k recommendations
                squared_errors.append((actual - predicted) ** 2)
            
            rmse = np.sqrt(np.mean(squared_errors)) if squared_errors else 0
            metrics['rmse'].append(rmse)
        
        # Calculate averages for all metrics
        return {
            'precision@k': np.mean(metrics['precision@k']),
            'recall@k': np.mean(metrics['recall@k']),
            'f1@k': np.mean(metrics['f1@k']),
            'ndcg@k': np.mean(metrics['ndcg@k']),
            'rmse': np.mean(metrics['rmse'])
        }


