import pandas as pd
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data.data_loader import MovieDataLoader
from models.collaborative_filtering import CollaborativeFiltering
from models.content_based import ContentBasedRecommender
from models.hypergraph import HypergraphRecommender
from models.Enhanced import EnhancedHypergraphRecommender

def prepare_test_set(imdb_format_movies):
    test_set = []
    for movie in imdb_format_movies:
        print(f"movie: {movie}")  # Debugging line to check the data structure
        if isinstance(movie, dict):
            movie_id = movie['movie_id']  # Now accessing the dictionary properly
            relevant_movie_ids = movie.get('relevant_movie_ids', [])  # Assuming this is how relevant movies are stored
            test_set.append((movie_id, relevant_movie_ids))
        else:
            print("Warning: Expected dictionary but got:", type(movie))
    return test_set

class MovieDataLoader:
    def load_tmdb_movies(self, tmdb_movies_path):
        return pd.read_csv(tmdb_movies_path)  # Load movies from CSV

    def load_tmdb_credits(self, tmdb_credits_path):
        return pd.read_csv(tmdb_credits_path)  # Load credits from CSV

    def get_relevant_movies(self, movie):
        # Placeholder function to fetch relevant movies. Adjust this as needed.
        return []  # Returning empty list for now, can be modified based on the dataset

    def convert_to_original_format(self, movies, credits):
        imdb_format_movies = []
        for idx, movie in movies.iterrows():  # Assuming movies is a DataFrame
            # Ensure each entry is a dictionary with 'movie_id' and 'relevant_movie_ids'
            imdb_format_movies.append({
                'movie_id': movie['id'],  # Adjust with actual movie ID column name
                'relevant_movie_ids': self.get_relevant_movies(movie)  # Example of relevant movies
            })
        return imdb_format_movies

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
    
    # Prepare the test set
    test_set = prepare_test_set(imdb_format_movies)
    
    # Initialize models
    cf_model = CollaborativeFiltering()
    content_model = ContentBasedRecommender()
    hypergraph_model = HypergraphRecommender(embedding_dim=64, n_layers=2)
    enhanced_hypergraph_model = EnhancedHypergraphRecommender(embedding_dim=64, n_layers=2, num_heads=2)
    
    # Fit models on training data (you need to define the `fit` method for each model)
    print("Fitting models...")
    cf_model.fit(imdb_format_movies)  # Assuming the fit method requires these data
    content_model.fit(imdb_format_movies)  # Similarly for content-based model
    hypergraph_model.fit(imdb_format_movies)  # Hypergraph model fit
    enhanced_hypergraph_model.fit(imdb_format_movies)  # Enhanced model fit
    
    # Evaluate all models
    models = {
        "Collaborative Filtering": cf_model,
        "Content-Based": content_model,
        "Hypergraph": hypergraph_model,
        "Enhanced Hypergraph": enhanced_hypergraph_model,
    }
    
    # Display model evaluation results
    print("Evaluating models...\n")
    results = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        # Call the evaluate_models function for each model
        result = model.evaluate_models(test_set, k=10)  # You can adjust k as needed
        results[model_name] = result
    
    # Display results
    print("\nModel Evaluation Results:")
    for model_name, result in results.items():
        print(f"\n{model_name} Evaluation:")
        for metric, value in result.items():
            print(f"{metric}: {value:.4f}")
        
    # Optionally, you can save results to a file or further process them.
    # For example, saving to a CSV:
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_evaluation_results.csv', index=False)
    
if __name__ == "__main__":
    main()
