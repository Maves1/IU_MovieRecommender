import torch
import numpy as np
import pandas as pd
import warnings

from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


class MatrixFactorization(torch.nn.Module):

    '''

    The MatrixFactorization class is designed for matrix factorization-based collaborative
    filtering using PyTorch. It learns embeddings for users and items in a matrix,
    enabling the prediction of user-item interactions.

    '''
    
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()

        # User embeddings (users to their features)
        self.user_factors = torch.nn.Embedding(n_users, n_factors)

        # Movie embeddings (movies to their features)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)
        self.user_factors.weight.data.uniform_(0, 0.05)
        self.item_factors.weight.data.uniform_(0, 0.05)
    
    def forward(self, data):
        users, movies = data[:, 0], data[:, 1]
        return (self.user_factors(users) * self.item_factors(movies)).sum(1)
    
    def predict(self, user):
        return self.forward(user)


# Data loader
class Loader(Dataset):

    '''

    The Loader class is a PyTorch Dataset used for handling rating data.
    It transforms the input ratings dataset into a format suitable for training
    machine learning models.

    '''

    def __init__(self, ratings_df):
        self.ratings = ratings_df.copy()
        
        # Obtaining all unique user and movie ids
        users = ratings_df["user_id"].unique()
        movies = ratings_df["movie_id"].unique()
        
        # We need to create mappings from unique vals to indices
        self.userid2idx = {o: i for i, o in enumerate(users)}
        self.movieid2idx = {o: i for i, o in enumerate(movies)}
        
        # Doing the opposite thing
        self.idx2userid = {i: o for o, i in self.userid2idx.items()}
        self.idx2movieid = {i: o for o, i in self.movieid2idx.items()}
        
        # We also need to replace initial ids with indices
        self.ratings["movie_id"] = ratings_df["movie_id"].apply(lambda x: self.movieid2idx[x])
        self.ratings["user_id"]= ratings_df["user_id"].apply(lambda x: self.userid2idx[x])
        
        self.x = self.ratings.drop(['rating', 'timestamp'], axis=1).values
        self.y = self.ratings['rating'].values
        self.x, self.y = torch.tensor(self.x), torch.tensor(self.y)

    # Return item by its index
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    # Return len of ratings
    def __len__(self):
        return len(self.ratings)

'''
get_recommended_movies() returns titles of recommended movies
'''
def get_recommended_movies(model, loader, user_id, user_rated_movies, recommended_count):
    movie_ids = np.array(list(loader.movieid2idx.values()))

    # We should not recommend movies that the user has already rated
    # so let's remove them
    needed_movies = np.setdiff1d(movie_ids, user_rated_movies)

    # Predicting
    predictions = model.predict(torch.tensor([[user_id, movie_id] for movie_id in needed_movies]))

    top_movie_indices = torch.argsort(predictions, descending=True)
    top_movie_indices = top_movie_indices[:recommended_count]  # we need to slice using the needed num of movies

    # Obtaining ids of the most recommended movies
    top_movie_ids = needed_movies[top_movie_indices]

    # We return ids of the movies
    return top_movie_ids
