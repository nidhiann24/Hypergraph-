import pandas as pd
import numpy as np
from typing import List, Tuple
from collections import defaultdict
import torch
import os
import json
from torch.nn import LayerNorm
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F

import sys
import os



# Add the recommender root directory to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.data_loader import MovieDataLoader

class AttentionLayer(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int = 4):
        super(AttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(feature_dim, num_heads)
        self.norm = LayerNorm(feature_dim)
        
    def forward(self, x: torch.Tensor):
        x_norm = self.norm(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        return x + attn_output

class EnhancedHypergraphLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        super(EnhancedHypergraphLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.attention = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(out_features)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.attention)
        
    def forward(self, X: torch.Tensor, H: torch.Tensor, D_v: torch.Tensor, D_e: torch.Tensor):
        D_v_inv = torch.inverse(D_v)
        D_e_inv = torch.inverse(D_e)
        
        # Enhanced feature transformation with dropout
        theta = self.dropout(torch.matmul(X, self.W))
        
        # Attention-based hypergraph convolution
        coeff = torch.matmul(torch.matmul(D_v_inv, H), D_e_inv)
        attention_scores = F.softmax(torch.matmul(theta, self.attention), dim=0)
        coeff = torch.matmul(coeff, H.t()) * attention_scores
        
        # Apply layer normalization
        output = self.norm(torch.matmul(coeff, theta))
        return output

class EnhancedHypergraphRecommender:
    def __init__(self, embedding_dim: int = 128, n_layers: int = 3, num_heads: int = 4, dropout: float = 0.2):
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.df = None
        self.hyperedge_dict = defaultdict(set)
        self.vertex_features = None
        self.H = None
        self.model = None
        self.D_v = None
        self.D_e = None
        
    def _create_enhanced_hyperedges(self):
        """Create more sophisticated hyperedges with additional features"""
        # Original hyperedges
        self._create_basic_hyperedges()
        
        # Rating-based hyperedges
        for idx, row in self.df.iterrows():
            if pd.notnull(row['vote_average']):
                rating_group = f"rating_{int(row['vote_average'] // 2)}"
                self.hyperedge_dict[rating_group].add(idx)
        
        # Popularity-based hyperedges
        if 'popularity' in self.df.columns:
            popularity_quantiles = pd.qcut(self.df['popularity'].fillna(0), q=5, labels=False)
            for idx, quantile in enumerate(popularity_quantiles):
                self.hyperedge_dict[f'popularity_group_{quantile}'].add(idx)
        
        # Combined genre-year hyperedges
        for idx, row in self.df.iterrows():
            if pd.notnull(row['genres']) and pd.notnull(row['title_year']):
                decade = (row['title_year'] // 10) * 10
                genres = row['genres'].split('|')
                for genre in genres:
                    self.hyperedge_dict[f'genre_decade_{genre}_{decade}'].add(idx)

    def _create_basic_hyperedges(self):
        """Create basic hyperedges from the original implementation"""
        for idx, row in self.df.iterrows():
            # Genre hyperedges
            if pd.notnull(row['genres']):
                for genre in row['genres'].split('|'):
                    self.hyperedge_dict[f'genre_{genre}'].add(idx)
            
            # Director and actor hyperedges
            if pd.notnull(row['director_name']):
                self.hyperedge_dict[f'director_{row["director_name"]}'].add(idx)
            
            for i in range(1, 4):
                actor_col = f'actor_{i}_name'
                if pd.notnull(row[actor_col]):
                    self.hyperedge_dict[f'actor_{row[actor_col]}'].add(idx)

    def _create_incidence_matrix(self):
        """Create the incidence matrix H from hyperedge dictionary"""
        num_vertices = len(self.df)
        num_edges = len(self.hyperedge_dict)
        H = torch.zeros((num_vertices, num_edges))
        
        for edge_idx, (_, vertices) in enumerate(self.hyperedge_dict.items()):
            for vertex_idx in vertices:
                H[vertex_idx, edge_idx] = 1.0
                
        return H

    def _create_degree_matrices(self, H):
        """Create the degree matrices D_v and D_e"""
        vertex_degrees = torch.sum(H, dim=1)
        edge_degrees = torch.sum(H, dim=0)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        D_v = torch.diag(vertex_degrees + epsilon)
        D_e = torch.diag(edge_degrees + epsilon)
        
        return D_v, D_e

    def _initialize_enhanced_features(self):
        """Initialize more comprehensive vertex features"""
        features = []
        
        for _, row in self.df.iterrows():
            feature_vec = []
            
            # Basic features
            feature_vec.extend([
                row['vote_average'] / 10.0 if pd.notnull(row['vote_average']) else 0,
                min(row['num_voted_users'] / 10000.0, 1.0) if pd.notnull(row['num_voted_users']) else 0,
            ])
            
            # Popularity feature
            if 'popularity' in self.df.columns:
                feature_vec.append(min(row['popularity'] / 100.0, 1.0) if pd.notnull(row['popularity']) else 0)
            
            # Budget and revenue features
            if 'budget' in self.df.columns and 'revenue' in self.df.columns:
                budget = row['budget'] if pd.notnull(row['budget']) else 0
                revenue = row['revenue'] if pd.notnull(row['revenue']) else 0
                feature_vec.extend([
                    min(budget / 1e8, 1.0),
                    min(revenue / 1e8, 1.0),
                    min(revenue / (budget + 1), 10.0) / 10.0  # ROI
                ])
            
            # Temporal features
            year = row['title_year'] if pd.notnull(row['title_year']) else 2000
            feature_vec.extend([
                (year - 1900) / 200.0,  # Normalized year
                np.sin(2 * np.pi * year / 100),  # Cyclical year encoding
                np.cos(2 * np.pi * year / 100)
            ])
            
            features.append(feature_vec)
            
        return torch.FloatTensor(features)

    def _build_model(self):
        """Build the enhanced neural network model"""
        input_dim = self.vertex_features.shape[1]
        
        layers = []
        for i in range(self.n_layers):
            # Add hypergraph convolution layer
            layers.append(
                EnhancedHypergraphLayer(
                    input_dim if i == 0 else self.embedding_dim,
                    self.embedding_dim,
                    self.dropout
                )
            )
            # Add attention layer
            layers.append(AttentionLayer(self.embedding_dim, self.num_heads))
            # Add non-linearity
            layers.append(nn.PReLU())
            
        self.model = nn.ModuleList(layers)

    def fit(self, df: pd.DataFrame, epochs: int = 50, lr: float = 0.001, weight_decay: float = 1e-5):
        """Fit the enhanced hypergraph recommender"""
        self.df = df.copy(deep=True)
        
        # Create enhanced hyperedges and matrices
        self._create_enhanced_hyperedges()
        self.H = self._create_incidence_matrix()
        self.D_v, self.D_e = self._create_degree_matrices(self.H)
        
        # Initialize enhanced features
        self.vertex_features = self._initialize_enhanced_features()
        
        # Build and train the model
        self._build_model()
        self._train_model(epochs, lr, weight_decay)

    def _train_model(self, epochs: int, lr: float, weight_decay: float):
        """Enhanced training procedure with additional optimization techniques"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in tqdm(range(epochs), desc="Training"):
            # Forward pass
            X = self.vertex_features
            for layer in self.model:
                X = layer(X, self.H, self.D_v, self.D_e) if isinstance(layer, EnhancedHypergraphLayer) else layer(X)
            
            # Compute enhanced loss
            adj = torch.matmul(self.H, self.H.t())
            pred = torch.matmul(X, X.t())
            reconstruction_loss = F.mse_loss(pred, adj)
            
            # Add L2 regularization
            l2_reg = torch.tensor(0., requires_grad=True)
            for param in self.model.parameters():
                l2_reg = l2_reg + torch.norm(param)
            
            loss = reconstruction_loss + 0.01 * l2_reg
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    def get_recommendations(self, movie_id: int, n_recommendations: int = 5) -> List[Tuple[str, float]]:
        """Get enhanced recommendations with confidence scores"""
        if movie_id >= len(self.df):
            return []
        
        with torch.no_grad():
            # Get embeddings
            X = self.vertex_features
            for layer in self.model:
                X = layer(X, self.H, self.D_v, self.D_e) if isinstance(layer, EnhancedHypergraphLayer) else layer(X)
            
            # Calculate similarities with temperature scaling
            movie_embedding = X[movie_id].unsqueeze(0)
            similarities = F.cosine_similarity(movie_embedding, X)
            similarities = similarities.numpy()
            
            # Get top recommendations
            top_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]
            
            # Format results with additional metadata
            recommendations = []
            for idx in top_indices:
                movie_data = self.df.iloc[idx]
                rec_info = (
                    movie_data['movie_title'],
                    float(similarities[idx])
                )
                recommendations.append(rec_info)
        
        return recommendations

    def save_model(self, save_dir: str):
        """Save the enhanced model and data"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'model.pt'))
        
        # Save other necessary data in chunks
        chunk_size = 10000
        df_chunks = [self.df.iloc[i:i + chunk_size] for i in range(0, len(self.df), chunk_size)]
        for i, chunk in enumerate(df_chunks):
            chunk.to_json(os.path.join(save_dir, f'df_chunk_{i}.json'), orient='split')
        
        state = {
            'embedding_dim': self.embedding_dim,
            'n_layers': self.n_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'num_chunks': len(df_chunks),
            'vertex_features': self.vertex_features.numpy().tolist(),
            'H': self.H.numpy().tolist(),
            'D_v': self.D_v.numpy().tolist(),
            'D_e': self.D_e.numpy().tolist(),
            'hyperedge_dict': {k: list(v) for k, v in self.hyperedge_dict.items()}
        }
        
        with open(os.path.join(save_dir, 'state.json'), 'w') as f:
            json.dump(state, f)
        
    @classmethod
    def load_model(cls, save_dir: str) -> 'EnhancedHypergraphRecommender':
        """Load the enhanced model"""
        try:
            with open(os.path.join(save_dir, 'state.json'), 'r') as f:
                state = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from state file: {e}")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        # Create new instance
        instance = cls(
            embedding_dim=state['embedding_dim'],
            n_layers=state['n_layers'],
            num_heads=state['num_heads'],
            dropout=state['dropout']
        )
        
        # Restore data
        df_chunks = []
        for i in range(state['num_chunks']):
            chunk = pd.read_json(os.path.join(save_dir, f'df_chunk_{i}.json'), orient='split')
            df_chunks.append(chunk)
        instance.df = pd.concat(df_chunks, ignore_index=True)
        instance.vertex_features = torch.FloatTensor(state['vertex_features'])
        instance.H = torch.FloatTensor(state['H'])
        instance.D_v = torch.FloatTensor(state['D_v'])
        instance.D_e = torch.FloatTensor(state['D_e'])
        instance.hyperedge_dict = defaultdict(set, {k: set(v) for k, v in state['hyperedge_dict'].items()})
        
        # Rebuild and load model
        instance._build_model()
        instance.model.load_state_dict(torch.load(os.path.join(save_dir, 'model.pt')))
        
        return instance
        
    def evaluate_model(self, test_df: pd.DataFrame, k: int = 10) -> dict:
        """Evaluate the model using precision, recall, and F1-score"""
        self.df = test_df.copy(deep=True)
        
        # Create enhanced hyperedges and matrices
        self._create_enhanced_hyperedges()
        self.H = self._create_incidence_matrix()
        self.D_v, self.D_e = self._create_degree_matrices(self.H)
        
        # Initialize enhanced features
        self.vertex_features = self._initialize_enhanced_features()
        
        # Get embeddings
        with torch.no_grad():
            X = self.vertex_features
            for layer in self.model:
                X = layer(X, self.H, self.D_v, self.D_e) if isinstance(layer, EnhancedHypergraphLayer) else layer(X)
            
            # Calculate precision, recall, and F1-score
            precision_list = []
            recall_list = []
            f1_list = []
            
            for movie_id in range(len(test_df)):
                true_positives = 0
                false_positives = 0
                false_negatives = 0
                
                recommendations = self.get_recommendations(movie_id, n_recommendations=k)
                recommended_ids = [self.df[self.df['movie_title'] == rec[0]].index[0] for rec in recommendations]
                
                for rec_id in recommended_ids:
                    if rec_id in self.hyperedge_dict[f'genre_{self.df.iloc[movie_id]["genres"].split("|")[0]}']:
                        true_positives += 1
                    else:
                        false_positives += 1
                
                false_negatives = len(self.hyperedge_dict[f'genre_{self.df.iloc[movie_id]["genres"].split("|")[0]}']) - true_positives
                
                precision = true_positives / (true_positives + false_positives + 1e-8)
                recall = true_positives / (true_positives + false_negatives + 1e-8)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
                
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
            
            metrics = {
                'precision': np.mean(precision_list),
                'recall': np.mean(recall_list),
                'f1_score': np.mean(f1_list)
            }
        
            return metrics

