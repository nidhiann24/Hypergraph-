import pandas as pd
import json
import numpy as np
from typing import Tuple, Dict, List

class MovieDataLoader:
    def __init__(self):
        self.TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES = {
            'budget': 'budget',
            'genres': 'genres',
            'revenue': 'gross',
            'title': 'movie_title',
            'runtime': 'duration',
            'original_language': 'language',
            'keywords': 'plot_keywords',
            'vote_count': 'num_voted_users'
        }

    def load_tmdb_movies(self, path: str) -> pd.DataFrame:
        """Load and preprocess TMDB movies dataset"""
        df = pd.read_csv(path)
        df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.date())
        json_columns = ['genres', 'keywords', 'production_countries',
                       'production_companies', 'spoken_languages']
        for column in json_columns:
            df[column] = df[column].apply(json.loads)
        return df

    def load_tmdb_credits(self, path: str) -> pd.DataFrame:
        """Load and preprocess TMDB credits dataset"""
        df = pd.read_csv(path)
        json_columns = ['cast', 'crew']
        for column in json_columns:
            df[column] = df[column].apply(json.loads)
        return df

    def safe_access(self, container, index_values):
        """Safely access nested container values"""
        result = container
        try:
            for idx in index_values:
                result = result[idx]
            return result
        except (IndexError, KeyError):
            return np.nan

    def get_director(self, crew_data: List) -> str:
        """Extract director name from crew data"""
        directors = [x['name'] for x in crew_data if x['job'] == 'Director']
        return self.safe_access(directors, [0])

    def pipe_flatten_names(self, keywords: List) -> str:
        """Convert list of keyword dictionaries to pipe-separated string"""
        return '|'.join([x['name'] for x in keywords])

    def convert_to_original_format(self, movies: pd.DataFrame, credits: pd.DataFrame) -> pd.DataFrame:
        """Convert TMDB format to IMDB format"""
        tmdb_movies = movies.copy()
        tmdb_movies.rename(columns=self.TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES, inplace=True)
        
        tmdb_movies['title_year'] = pd.to_datetime(tmdb_movies['release_date']).apply(lambda x: x.year)
        tmdb_movies['country'] = tmdb_movies['production_countries'].apply(lambda x: self.safe_access(x, [0, 'name']))
        tmdb_movies['language'] = tmdb_movies['spoken_languages'].apply(lambda x: self.safe_access(x, [0, 'name']))
        
        # Add director and actor information
        tmdb_movies['director_name'] = credits['crew'].apply(self.get_director)
        tmdb_movies['actor_1_name'] = credits['cast'].apply(lambda x: self.safe_access(x, [1, 'name']))
        tmdb_movies['actor_2_name'] = credits['cast'].apply(lambda x: self.safe_access(x, [2, 'name']))
        tmdb_movies['actor_3_name'] = credits['cast'].apply(lambda x: self.safe_access(x, [3, 'name']))
        
        # Process genres and keywords
        tmdb_movies['genres'] = tmdb_movies['genres'].apply(self.pipe_flatten_names)
        tmdb_movies['plot_keywords'] = tmdb_movies['plot_keywords'].apply(self.pipe_flatten_names)
        
        return tmdb_movies
