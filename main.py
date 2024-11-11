import pandas as pd
import os
import sys

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from data.data_loader import MovieDataLoader
from models.collaborative_filtering import CollaborativeFiltering
from models.content_based import ContentBasedRecommender
from models.hypergraph import HypergraphRecommender
from models.Enhanced import EnhancedHypergraphRecommender

def collaborative_filtering_recommendation(imdb_format_movies):
    """Run collaborative filtering recommendation with normalized similarity score."""
    cf_model = CollaborativeFiltering()
    cf_model.fit(imdb_format_movies)
    
    movie_title = input("Enter a movie title to get collaborative filtering recommendations: ")
    recommendations = cf_model.get_recommendations(movie_title, n_recommendations=5)
    
    if recommendations:
        # Extract the similarity scores from the recommendations
        scores = [score for _, score in recommendations]
        
        # Normalize the similarity scores between 0 and 1
        min_score = min(scores)
        max_score = max(scores)
        normalized_scores = [(score - min_score) / (max_score - min_score) if max_score > min_score else 1.0 for score in scores]
        
        print("\nTop 5 movie recommendations (Collaborative Filtering):")
        for i, (rec_movie, score) in enumerate(recommendations, start=1):
            # Get the normalized score from the list
            normalized_score = normalized_scores[i-1]
            print(f"{i}. {rec_movie} (Normalized similarity score: {normalized_score:.2f})")
    else:
        print("Sorry, no recommendations found for the given movie title.")


def content_based_recommendation(imdb_format_movies):
    """Run content-based recommendation using movie title."""
    content_model = ContentBasedRecommender()
    content_model.fit(imdb_format_movies)
    movie_title = input("Enter a movie title to get content-based recommendations: ")
    
    # Find movie index by title
    movie_id = imdb_format_movies.index[
        imdb_format_movies['movie_title'].str.lower() == movie_title.lower()
    ].tolist()
    
    if not movie_id:
        print("Movie not found in the dataset. Please try another title.")
        return

    movie_id = movie_id[0]  # Get the first match
    recommendations = content_model.get_recommendations(movie_id, n_recommendations=5)
    
    if recommendations:
        print("\nTop 5 movie recommendations (Content-Based):")
        for i, (rec_movie, score) in enumerate(recommendations, start=1):
            print(f"{i}. {rec_movie} (Similarity score: {score:.3f})")
    else:
        print("Sorry, no recommendations found for the given movie.")

def get_or_train_hypergraph_model(imdb_format_movies, model_dir='saved_models/hypergraph'):
    """Get a trained hypergraph model, loading from disk if available or training if needed."""
    if os.path.exists(model_dir):
        print("Loading pre-trained hypergraph model...")
        try:
            return HypergraphRecommender.load_model(model_dir)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
    else:
        print("No pre-trained model found. Training new hypergraph model (this may take a few minutes)...")
    
    # Train new model
    hypergraph_model = HypergraphRecommender(embedding_dim=64, n_layers=2)
    try:
        hypergraph_model.fit(imdb_format_movies)
        
        # Save the trained model
        print("Saving trained model...")
        os.makedirs(model_dir, exist_ok=True)
        hypergraph_model.save_model(model_dir)
        print("Model saved successfully!")
        
        return hypergraph_model
    except Exception as e:
        print(f"Error training hypergraph model: {str(e)}")
        return None

def get_or_train_enhanced_hypergraph_model(imdb_format_movies, model_dir='saved_models/enhanced_hypergraph'):
    """Get a trained enhanced hypergraph model, loading from disk if available or training if needed."""
    if os.path.exists(model_dir):
        print("Loading pre-trained enhanced hypergraph model...")
        try:
            return EnhancedHypergraphRecommender.load_model(model_dir)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
    else:
        print("No pre-trained model found. Training new enhanced hypergraph model (this may take a few minutes)...")
    
    # Train new model with reduced parameters
    enhanced_model = EnhancedHypergraphRecommender(
        embedding_dim=64,  # Reduced dimension
        n_layers=2,       # Reduced layers
        num_heads=2       # Reduced attention heads
    )
    
    try:
        # Monitor memory usage
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Initial memory usage: {initial_memory:.2f} MB")
        
        # Train with progress tracking
        print("Starting model training...")
        enhanced_model.fit(imdb_format_movies, epochs=5)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Final memory usage: {final_memory:.2f} MB")
        print(f"Memory difference: {final_memory - initial_memory:.2f} MB")
        
        # Save the trained model
        print("Saving trained model...")
        os.makedirs(model_dir, exist_ok=True)
        enhanced_model.save_model(model_dir)
        print("Model saved successfully!")
        
        return enhanced_model
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Cleaning up...")
        if os.path.exists(model_dir):
            import shutil
            shutil.rmtree(model_dir)
        return None
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def hypergraph_recommendation(imdb_format_movies, hypergraph_model=None):
    """Run hypergraph-based recommendation using movie title."""
    if hypergraph_model is None:
        hypergraph_model = get_or_train_hypergraph_model(imdb_format_movies)
        if hypergraph_model is None:
            return
    
    movie_title = input("Enter a movie title to get hypergraph-based recommendations: ")
    
    # Find movie index by title
    movie_id = imdb_format_movies.index[
        imdb_format_movies['movie_title'].str.lower() == movie_title.lower()
    ].tolist()
    
    if not movie_id:
        print("Movie not found in the dataset. Please try another title.")
        return

    movie_id = movie_id[0]  # Get the first match
    
    try:
        recommendations = hypergraph_model.get_recommendations(movie_id, n_recommendations=5)
        
        if recommendations:
            # Normalize similarity scores using min-max scaling
            scores = [score for _, score in recommendations]
            min_score = min(scores)
            max_score = max(scores)
            normalized_scores = [(score - min_score) / (max_score - min_score) if max_score > min_score else 1.0 for score in scores]
            
            '''# Avoid division by zero if all scores are the same
            if max_score == min_score:
                normalized_scores = [0.5 for _ in scores]  # Assign a default normalized score if no variation
            else:
                normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]'''
            
            print("\nTop 5 movie recommendations (Hypergraph-Based):")
            for i, ((rec_movie, _), norm_score) in enumerate(zip(recommendations, normalized_scores), start=1):
                print(f"{i}. {rec_movie} (Similarity score: {norm_score:.2f})")
        else:
            print("Sorry, no recommendations found for the given movie.")
    except Exception as e:
        print(f"Error getting recommendations: {str(e)}")


def enhanced_hypergraph_recommendation(imdb_format_movies, enhanced_model=None):
    """Run enhanced hypergraph-based recommendation using movie title."""
    if enhanced_model is None:
        enhanced_model = get_or_train_enhanced_hypergraph_model(imdb_format_movies)
        if enhanced_model is None:
            return
    
    movie_title = input("Enter a movie title to get enhanced hypergraph-based recommendations: ")
    
    # Find movie index by title
    movie_id = imdb_format_movies.index[
        imdb_format_movies['movie_title'].str.lower() == movie_title.lower()
    ].tolist()
    
    if not movie_id:
        print("Movie not found in the dataset. Please try another title.")
        return

    movie_id = movie_id[0]  # Get the first match
    
    try:
        recommendations = enhanced_model.get_recommendations(movie_id, n_recommendations=5)
        
        if recommendations:
            # Extract scores for normalization
            scores = [score for _, score in recommendations]
            min_score = min(scores)
            max_score = max(scores)
            
            # Normalize scores using min-max scaling
            if max_score == min_score:
                normalized_scores = [0.5 for _ in scores]  # Default value if all scores are identical
            else:
                normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
            
            print("\nTop 5 movie recommendations (Enhanced Hypergraph):")
            for i, ((rec_movie, _), norm_score) in enumerate(zip(recommendations, normalized_scores), start=1):
                print(f"{i}. {rec_movie} (Normalized Similarity score: {norm_score:.2f})")
        else:
            print("Sorry, no recommendations found for the given movie.")
    except Exception as e:
        print(f"Error getting recommendations: {str(e)}")

def display_movie_stats(movies_df):
    """Display basic statistics about the movie dataset."""
    print("\nDataset Statistics:")
    print(f"Total number of movies: {len(movies_df)}")
    print(f"Year range: {movies_df['title_year'].min():.0f} - {movies_df['title_year'].max():.0f}")
    print(f"Average vote score: {movies_df['vote_average'].mean():.2f}")
    print(f"Average number of votes: {movies_df['num_voted_users'].mean():.0f}")
    
    # Display top 5 genres
    genres = []
    for genre_list in movies_df['genres'].dropna():
        genres.extend(genre_list.split('|'))
    genre_counts = pd.Series(genres).value_counts()
    print("\nTop 5 genres in the dataset:")
    for genre, count in genre_counts.head().items():
        print(f"{genre}: {count} movies")

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
    
    # Display dataset statistics
    display_movie_stats(imdb_format_movies)
    
    #Initialize both hypergraph models once
    hypergraph_model = get_or_train_hypergraph_model(imdb_format_movies)
    enhanced_hypergraph_model = get_or_train_enhanced_hypergraph_model(imdb_format_movies)
    
    while True:
        # Display the recommendation menu
        print("\nMovie Recommendation Options")
        print("1. Get Collaborative Filtering Recommendations")
        print("2. Get Content-Based Recommendations")
        print("3. Get Hypergraph-Based Recommendations")
        print("4. Get Enhanced Hypergraph Recommendations")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ")
        
        match choice:
            case "1":
                collaborative_filtering_recommendation(imdb_format_movies)
            case "2":
                content_based_recommendation(imdb_format_movies)
            case "3":
                hypergraph_recommendation(imdb_format_movies, hypergraph_model)
            case "4":
                enhanced_hypergraph_recommendation(imdb_format_movies, enhanced_hypergraph_model)
            case "5":
                print("Thank you for using the movie recommendation system!")
                break
            case _:
                print("Invalid choice. Please enter a number between 1 and 5.")
        
        # Ask if user wants to continue
        if input("\nWould you like to try another recommendation? (y/n): ").lower() != 'y':
            print("Thank you for using the movie recommendation system!")
            break

if __name__ == "__main__":
    main()
