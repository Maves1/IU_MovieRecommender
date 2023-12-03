import torch
import numpy as np
import pandas as pd

import os
import sys

# some hacks to import the model
sys.path.append(os.path.abspath(os.path.dirname(__file__)).replace("/benchmark", ""))

from models.model import MatrixFactorization, Loader, get_recommended_movies


# Let's calculate some metrics
def calculate_rmse(model, loader):
    pass

test_model = torch.load("./models/supermodel.pth")

# Dataset root folder
dataset_root_path = "./data/interim/ml-100k"

u_data_df = pd.read_csv(f"{dataset_root_path}/u.data", delimiter="\t", names=["user_id", "movie_id", "rating", "timestamp"], header=None)

u_item_df = pd.read_csv(f"{dataset_root_path}/u.item", delimiter="|", names=["movie_id", "movie_title", "release_date",
                                                                              "video_release_date", "IMDb_URL",
                                                                              "unknown", "Action", "Adventure", "Animation", "Childrens",
                                                                              "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                                                                              "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
                                                                              "War", "Western"], encoding="cp1252")

train_set = Loader(u_data_df)
movie_titles = u_item_df.set_index("movie_id")["movie_title"].to_dict()

# Let's recommend 3 movies
user_id = 45
user_movies = u_data_df['user_id'] == user_id
user_ratings = u_data_df[user_movies]['movie_id'].values

recommended_movie_ids = get_recommended_movies(test_model, train_set, user_id, user_ratings, 5)
recommended_movie_titles = [movie_titles[movie_id] for movie_id in recommended_movie_ids]

rated_count_show = 8
# Let's compare movies that the user has rated and recommended movies
print("Movies rated by the user:")
for _, movie_id in enumerate(user_ratings[:rated_count_show]):
    print(movie_titles[movie_id])

print("\nMovies recommended to the user:")
for _, movie_title in enumerate(recommended_movie_titles):
    print(movie_title)
