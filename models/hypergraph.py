import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Set
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json

class HypergraphLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(HypergraphLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        
    def forward(self, X: torch.Tensor, H: torch.Tensor, D_v: torch.Tensor, D_e: torch.Tensor):
        """
        Hypergraph convolution operation
        X: Node features
        H: Incidence matrix
        D_v: Vertex degree matrix
        D_e: Hyperedge degree matrix
        """
        D_v_inv = torch.inverse(D_v)
        D_e_inv = torch.inverse(D_e)
        
        # Hypergraph convolution
        theta = torch.matmul(X, self.W)  # Feature transformation
        coeff = torch.matmul(torch.matmul(D_v_inv, H), D_e_inv)
        coeff = torch.matmul(coeff, H.t())
        
        return torch.matmul(coeff, theta)

class HypergraphRecommender:
    """
    HypergraphRecommender is a class that implements a hypergraph-based recommendation system for movies. 
    It uses movie attributes to create hyperedges and builds a hypergraph neural network to learn embeddings 
    for movie recommendations.
    Attributes:
        embedding_dim (int): Dimension of the embedding vectors.
        n_layers (int): Number of layers in the hypergraph neural network.
        df (pd.DataFrame): DataFrame containing movie data.
        hyperedge_dict (defaultdict): Dictionary mapping hyperedge types to sets of movie indices.
        vertex_features (torch.Tensor): Tensor containing vertex features.
        H (torch.Tensor): Incidence matrix of the hypergraph.
        model (nn.ModuleList): Hypergraph neural network model.
        D_v (torch.Tensor): Vertex degree matrix.
        D_e (torch.Tensor): Hyperedge degree matrix.
    Methods:
        __init__(embedding_dim: int = 64, n_layers: int = 2):
            Initializes the HypergraphRecommender with the specified embedding dimension and number of layers.
        _create_hyperedges():
            Creates different types of hyperedges based on movie attributes.
        _create_incidence_matrix() -> torch.Tensor:
            Creates the incidence matrix H.
        _create_degree_matrices(H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            Creates vertex and hyperedge degree matrices.
        _initialize_features() -> torch.FloatTensor:
            Initializes vertex features using movie attributes.
        fit(df: pd.DataFrame):
            Fits the hypergraph recommender with the dataset.
        _train_model(n_epochs: int = 50, lr: float = 0.01):
            Trains the hypergraph neural network.
        get_recommendations(movie_id: int, n_recommendations: int = 5) -> List[Tuple[str, float]]:
            Gets movie recommendations based on hypergraph embeddings.
        save_model(save_dir: str):
            Saves the model and necessary data to disk.
        load_model(cls, save_dir: str) -> 'HypergraphRecommender':
            Loads a saved model and returns a new HypergraphRecommender instance.
        evaluate_model(test_pairs: List[Tuple[int, int]], k: int = 10) -> Dict[str, float]:
            Evaluates the model using various metrics.
    """
    def __init__(self, embedding_dim: int = 64, n_layers: int = 2):
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.df = None
        self.hyperedge_dict = defaultdict(set)
        self.vertex_features = None
        self.H = None  # Incidence matrix
        self.model = None
        self.D_v = None
        self.D_e = None
        
    def _create_hyperedges(self):
        """Create different types of hyperedges based on movie attributes"""
        # Genre hyperedges
        for idx, row in self.df.iterrows():
            if pd.notnull(row['genres']):
                genres = row['genres'].split('|')
                for genre in genres:
                    self.hyperedge_dict[f'genre_{genre}'].add(idx)
        
        # Director hyperedges
        for idx, row in self.df.iterrows():
            if pd.notnull(row['director_name']):
                self.hyperedge_dict[f'director_{row["director_name"]}'].add(idx)
        
        # Actor hyperedges
        for idx, row in self.df.iterrows():
            for i in range(1, 4):
                actor_col = f'actor_{i}_name'
                if pd.notnull(row[actor_col]):
                    self.hyperedge_dict[f'actor_{row[actor_col]}'].add(idx)
        
        # Year hyperedges (group movies by decade)
        for idx, row in self.df.iterrows():
            if pd.notnull(row['title_year']):
                decade = (row['title_year'] // 10) * 10
                self.hyperedge_dict[f'decade_{decade}'].add(idx)
        
    def _create_incidence_matrix(self):
        """Create the incidence matrix H"""
        n_vertices = len(self.df)
        n_edges = len(self.hyperedge_dict)
        
        # Initialize sparse incidence matrix
        H = torch.zeros((n_vertices, n_edges))
        
        # Fill incidence matrix
        for edge_idx, (edge_name, vertices) in enumerate(self.hyperedge_dict.items()):
            for vertex_idx in vertices:
                H[vertex_idx, edge_idx] = 1
                
        return H
    
    def _create_degree_matrices(self, H: torch.Tensor):
        """Create vertex and hyperedge degree matrices"""
        # Vertex degree matrix
        D_v = torch.diag(torch.sum(H, dim=1))
        
        # Hyperedge degree matrix
        D_e = torch.diag(torch.sum(H, dim=0))
        
        return D_v, D_e
    
    def _initialize_features(self):
        """Initialize vertex features using movie attributes"""
        features = []
        
        for _, row in self.df.iterrows():
            feature_vec = []
            
            # Add normalized vote average
            vote_avg = row['vote_average'] if pd.notnull(row['vote_average']) else 0
            feature_vec.append(vote_avg / 10.0)
            
            # Add normalized number of votes
            num_votes = row['num_voted_users'] if pd.notnull(row['num_voted_users']) else 0
            feature_vec.append(min(num_votes / 10000.0, 1.0))
            
            # Add decade indicator
            year = row['title_year'] if pd.notnull(row['title_year']) else 2000
            decade = (year // 10) * 10
            decade_vec = [0] * 13  # For decades from 1900s to 2020s
            decade_idx = min(max(int((decade - 1900) / 10), 0), 12)
            decade_vec[decade_idx] = 1
            feature_vec.extend(decade_vec)
            
            features.append(feature_vec)
            
        return torch.FloatTensor(features)
    
    def fit(self, df: pd.DataFrame):
        """Fit the hypergraph recommender with the dataset"""
        self.df = df.copy(deep=True)
        
        # Create hyperedges and incidence matrix
        self._create_hyperedges()
        self.H = self._create_incidence_matrix()
        
        # Create degree matrices
        self.D_v, self.D_e = self._create_degree_matrices(self.H)
        
        # Initialize vertex features
        self.vertex_features = self._initialize_features()
        
        # Initialize model
        input_dim = self.vertex_features.shape[1]
        self.model = nn.ModuleList([
            HypergraphLayer(input_dim if i == 0 else self.embedding_dim, self.embedding_dim)
            for i in range(self.n_layers)
        ])
        
        # Train the model
        self._train_model()
        
    def _train_model(self, n_epochs: int = 50, lr: float = 0.01):
        """Train the hypergraph neural network"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(n_epochs):
            # Forward pass
            X = self.vertex_features
            for layer in self.model:
                X = F.relu(layer(X, self.H, self.D_v, self.D_e))
            
            # Compute loss (reconstruction loss)
            adj = torch.matmul(self.H, self.H.t())
            pred = torch.matmul(X, X.t())
            loss = F.mse_loss(pred, adj)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
    
    def get_recommendations(self, movie_id: int, n_recommendations: int = 5) -> List[Tuple[str, float]]:
        """Get movie recommendations based on hypergraph embeddings"""
        if movie_id >= len(self.df):
            return []
        
        # Get embeddings
        with torch.no_grad():
            X = self.vertex_features
            for layer in self.model:
                X = F.relu(layer(X, self.H, self.D_v, self.D_e))
        
        # Calculate similarities
        movie_embedding = X[movie_id].unsqueeze(0)
        similarities = F.cosine_similarity(movie_embedding, X).numpy()
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]
        
        # Format results
        recommendations = [
            (self.df.iloc[idx]['movie_title'], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return recommendations

    def save_model(self, save_dir: str):
        """
        Save the model and necessary data to disk
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model state
        model_state = {
            'state_dict': self.model.state_dict(),
            'embedding_dim': self.embedding_dim,
            'n_layers': self.n_layers
        }
        torch.save(model_state, os.path.join(save_dir, 'model.pt'))
        
        # Save tensors
        torch.save({
            'H': self.H,
            'D_v': self.D_v,
            'D_e': self.D_e,
            'vertex_features': self.vertex_features
        }, os.path.join(save_dir, 'tensors.pt'))
        
        # Save DataFrame
        self.df.to_pickle(os.path.join(save_dir, 'dataframe.pkl'))
        
        # Save hyperedge dictionary
        hyperedge_dict_serializable = {k: list(v) for k, v in self.hyperedge_dict.items()}
        with open(os.path.join(save_dir, 'hyperedge_dict.json'), 'w') as f:
            json.dump(hyperedge_dict_serializable, f)

    @classmethod
    def load_model(cls, save_dir: str) -> 'HypergraphRecommender':
        """
        Load a saved model and return a new HypergraphRecommender instance
        """
        # Load model state and config
        model_state = torch.load(os.path.join(save_dir, 'model.pt'))
        
        # Create new instance
        recommender = cls(
            embedding_dim=model_state['embedding_dim'],
            n_layers=model_state['n_layers']
        )
        
        # Load tensors
        tensors = torch.load(os.path.join(save_dir, 'tensors.pt'))
        recommender.H = tensors['H']
        recommender.D_v = tensors['D_v']
        recommender.D_e = tensors['D_e']
        recommender.vertex_features = tensors['vertex_features']
        
        # Load DataFrame
        recommender.df = pd.read_pickle(os.path.join(save_dir, 'dataframe.pkl'))
        
        # Load hyperedge dictionary
        with open(os.path.join(save_dir, 'hyperedge_dict.json'), 'r') as f:
            hyperedge_dict_data = json.load(f)
            recommender.hyperedge_dict = defaultdict(set)
            for k, v in hyperedge_dict_data.items():
                recommender.hyperedge_dict[k] = set(v)
        
        # Initialize and load model
        input_dim = recommender.vertex_features.shape[1]
        recommender.model = nn.ModuleList([
            HypergraphLayer(input_dim if i == 0 else recommender.embedding_dim, recommender.embedding_dim)
            for i in range(recommender.n_layers)
        ])
        recommender.model.load_state_dict(model_state['state_dict'])
        
        return recommender
    def evaluate_model(self, test_pairs: List[Tuple[int, int]], k: int = 10) -> Dict[str, float]:
            """
            Evaluate the model using various metrics
            Args:
                test_pairs: List of (movie_id1, movie_id2) tuples representing ground truth similar pairs
                k: Number of recommendations to consider for metrics
            Returns:
                Dictionary containing evaluation metrics
            """
            precision_at_k = []
            recall_at_k = []
            ndcg_at_k = []

            for movie_id1, movie_id2 in test_pairs:
                # Get recommendations for movie1
                recs = self.get_recommendations(movie_id1, k)
                rec_ids = [self.df[self.df['movie_title'] == rec[0]].index[0] for rec in recs]
                
                # Calculate metrics
                relevant_items = set([movie_id2])
                retrieved_items = set(rec_ids)
                
                # Precision@K
                precision = len(relevant_items.intersection(retrieved_items)) / len(retrieved_items)
                precision_at_k.append(precision)
                
                # Recall@K
                recall = len(relevant_items.intersection(retrieved_items)) / len(relevant_items)
                recall_at_k.append(recall)
                
                # NDCG@K
                dcg = 0
                idcg = 1  # Ideal DCG for one relevant item
                for i, rec_id in enumerate(rec_ids):
                    if rec_id in relevant_items:
                        dcg += 1 / np.log2(i + 2)
                ndcg = dcg / idcg
                ndcg_at_k.append(ndcg)

            return {
                f'precision@{k}': np.mean(precision_at_k),
                f'recall@{k}': np.mean(recall_at_k),
                f'ndcg@{k}': np.mean(ndcg_at_k)
            }