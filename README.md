# Movie Recommender System

This repository contains a movie recommender system based on collaborative filtering
## Overview

The model predicts user-item interactions by learning embeddings for users and movies. Leveraging latent factors, it offers recommendations based on historical user ratings in the MovieLens dataset.

## Results

- **RMSE**: 1.4752
    - The model achieved an RMSE of 1.4752, indicating its accuracy in predicting user-item interactions.

## Usage

1. **Installation**:
    - Install necessary dependencies by running:
        `pip install -r requirements.txt`
2. **Running Benchmark**:
    - To evaluate the model, run:
        `python benchmark/model_test.py`
    - This script calculates RMSE and provides an example of model recommendations