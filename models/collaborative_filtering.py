import pandas as pd
import numpy as np
import sys
import os



# Add the recommender root directory to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.data_loader import MovieDataLoader

from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from sklearn.metrics import mean_squared_error
from math import sqrt


class CollaborativeFiltering:
    """A collaborative filtering recommendation system based on movie ratings and vote counts.
    This class implements a collaborative filtering approach using cosine similarity between movies
    based on their user vote counts and average ratings. It provides methods for training the model,
    generating recommendations, and evaluating the model's performance using various metrics.
    Attributes:
        df (pd.DataFrame): The input dataset containing movie ratings and votes.
        similarity_matrix (pd.DataFrame): Matrix containing cosine similarities between movies.
        title_case_map (dict): Mapping of lowercase movie titles to their proper case versions.
    Example:
        >>> cf = CollaborativeFiltering()
        >>> cf.fit(movies_df)
        >>> recommendations = cf.get_recommendations("The Dark Knight", n_recommendations=5)
        >>> metrics = cf.evaluate(test_df, k=5)
    Notes:
        - The input DataFrame should contain columns: 'movie_title', 'num_voted_users', 'vote_average'
        - The similarity matrix is computed using both vote counts and average ratings
        - Case-insensitive movie title matching is supported through title_case_map
    """
    def __init__(self):
        self.df = None
        self.similarity_matrix = None
        self.title_case_map = {}  # Add a mapping for case-insensitive lookup
        
    def fit(self, df: pd.DataFrame):
        """Fit the collaborative filtering model"""
        self.df = df.copy(deep=True)
        
        # Create user-item matrix using vote counts and scores
        user_item_matrix = self.df.pivot_table(
            index='movie_title',
            values=['num_voted_users', 'vote_average'],
            aggfunc='mean'
        ).fillna(0)
        
        # Create case-insensitive title mapping
        self.title_case_map = {title.lower(): title for title in user_item_matrix.index}
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(user_item_matrix)
        self.similarity_matrix = pd.DataFrame(
            self.similarity_matrix,
            index=user_item_matrix.index,
            columns=user_item_matrix.index
        )
        
    def get_recommendations(self, movie_title: str, n_recommendations: int = 5) -> List[Tuple[str, float]]:
        """Get movie recommendations based on collaborative filtering"""
        # Convert input title to lowercase for matching
        movie_title_lower = movie_title.lower()
        
        # Look up the proper case version of the title
        proper_title = self.title_case_map.get(movie_title_lower)
        
        if proper_title is None:
            return []
            
        # Get similarity scores using the proper case title
        movie_similarities = self.similarity_matrix[proper_title]
        
        # Sort by similarity and get top recommendations
        similar_movies = movie_similarities.sort_values(ascending=False)[1:n_recommendations+1]
        
        # Format results
        recommendations = [
            (movie, score) 
            for movie, score in similar_movies.items()
        ]
        
        return recommendations

    def evaluate_model(self, test_df: pd.DataFrame, k: int = 5) -> dict:
        """Modified evaluation method with fixed NDCG calculation"""
        if self.df is None or self.similarity_matrix is None:
            raise ValueError("Model has not been fitted yet.")
        
        test_user_item_matrix = test_df.pivot_table(
            index='movie_title',
            values=['num_voted_users', 'vote_average'],
            aggfunc='mean'
        ).fillna(0)
        
        mse_total = 0
        ndcg_total = 0
        precision_total = 0
        recall_total = 0
        count = 0
        
        for movie_title in test_user_item_matrix.index:
            if movie_title in self.similarity_matrix.index:
                # Get recommendations
                recommendations = self.get_recommendations(movie_title, k)
                recommended_titles = [rec[0] for rec in recommendations]
                
                # Get actual ratings for recommended movies
                actual_ratings = []
                for rec_title in recommended_titles:
                    if rec_title in test_user_item_matrix.index:
                        actual_ratings.append(test_user_item_matrix.loc[rec_title, 'vote_average'])
                    else:
                        actual_ratings.append(0)
                
                # Calculate metrics
                if len(actual_ratings) > 0:
                    # MSE
                    predicted_ratings = [rec[1] * 10 for rec in recommendations]  # Scale similarity to 0-10
                    mse = np.mean([(a - p) ** 2 for a, p in zip(actual_ratings, predicted_ratings)])
                    mse_total += mse
                    
                    # NDCG
                    ideal_ratings = sorted(actual_ratings, reverse=True)
                    dcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(actual_ratings))
                    idcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_ratings))
                    ndcg = dcg / idcg if idcg > 0 else 0
                    ndcg_total += ndcg
                    
                    # Precision & Recall
                    relevant_items = set(test_user_item_matrix[test_user_item_matrix['vote_average'] >= 7].index)
                    recommended_items = set(recommended_titles)
                    true_positives = len(relevant_items & recommended_items)
                    
                    precision = true_positives / len(recommended_items) if recommended_items else 0
                    recall = true_positives / len(relevant_items) if relevant_items else 0
                    
                    precision_total += precision
                    recall_total += recall
                    
                    count += 1
        
        # Calculate final metrics
        if count > 0:
            rmse = np.sqrt(mse_total / count)
            ndcg = ndcg_total / count
            precision = precision_total / count
            recall = recall_total / count
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                'RMSE': rmse,
                'NDCG@K': ndcg,
                'Precision@K': precision,
                'Recall@K': recall,
                'F1-Score': f1_score
            }
        else:
            return {
                'RMSE': float('inf'),
                'NDCG@K': 0,
                'Precision@K': 0,
                'Recall@K': 0,
                'F1-Score': 0
            }
def main():
    # Paths to data files
    tmdb_credits_path = r'D:\Recommender\Recommendation-System\data\tmdb_5000_credits.csv'  
    tmdb_movies_path = r'D:\Recommender\Recommendation-System\data\tmdb_5000_movies.csv'  
    
    # Initialize the data loader and load the data
    print("Loading movie data...")
    data_loader = MovieDataLoader()
    movies = data_loader.load_tmdb_movies(tmdb_movies_path)
    credits = data_loader.load_tmdb_credits(tmdb_credits_path)
    
    # Convert the loaded data to the desired format
    imdb_format_movies = data_loader.convert_to_original_format(movies, credits)
    
    
    cf_model = CollaborativeFiltering()
    cf_model.fit(imdb_format_movies)
    
    # Evaluate model on a sample dataset (using same data for simplicity here)
    print("\nEvaluating Collaborative Filtering Model:")
    metrics = cf_model.evaluate_model(imdb_format_movies, k=5)
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

   

if __name__ == "__main__":
    main()
