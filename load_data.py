import numpy as np
import pandas as pd
import wget
import os
import zipfile

def load_small():
    if not os.path.isdir('ml-latest-small'): # if directory not present then download and unzip
        url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
        filename = wget.download(url)
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(".")
            
    ratings_filename = 'ml-latest-small/ratings.csv'
    movie_filename = 'ml-latest-small/movies.csv'
    data = pd.read_csv(ratings_filename)
    movies = pd.read_csv(movie_filename)
    return data, movies

def load_medium():
    if not os.path.isdir('ml-10M100K'):
        url = 'http://files.grouplens.org/datasets/movielens/ml-10m.zip'
        filename = wget.download(url)
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(".")
        
    ratings_filename = 'ml-10M100K/ratings.dat'
    movie_filename = 'ml-10M100K/movies.dat'
    data = pd.read_csv(ratings_filename, sep='::', header=None, engine='python')
    data.columns = ['userId','movieId','rating','timestamp']
    movies = pd.read_csv(movie_filename, sep='::', header=None, engine='python')
    movies.columns = ['movieId','title','genre']
    return data, movies

def load_large():
    if not os.path.isdir('ml-25m'):
        url = 'http://files.grouplens.org/datasets/movielens/ml-25m.zip'
        filename = wget.download(url)
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(".")
            
    ratings_filename = 'ml-25m/ratings.csv'
    movie_filename = 'ml-25m/movies.csv'
    data = pd.read_csv(ratings_filename)
    movies = pd.read_csv(movie_filename)
    return data, movies

def load_dataset(size = 'small'):
    if size == 'large':
        data, movies = load_large()
    elif size == 'medium':
        data, movies = load_medium()
    else:
        data, movies = load_small()
    
    return data, movies

def convert(data, num_movies):
    '''
    Convert the data from RAW csv to a single vector for every user
    This vector has ids of rated movies by a user
    '''
    users = data['userId'].unique()
    data = data.values
    N = data.shape[0]
    movie_arr = []
    rating_arr = []
    i = 0
    index = 0
    
    for id in users:
        movie_ids = []
        user_rating = []
        while index < N and data[index][0] == id:
            movie_ids.append(data[index][1])
            user_rating.append(data[index][2]/5)
            # arr[i, data[index][1]] = data[index][2]/5
            index += 1
        
        movie_arr.append(list(map(int,movie_ids)))
        rating_arr.append(user_rating)
        i += 1
    
    movie_arr = np.array(movie_arr, dtype = object)
    rating_arr = np.array(rating_arr, dtype = object)
    
    return movie_arr, rating_arr

def pre_process(data, movies):
    '''
    Parameters
    ----------
    data : ratings dataframe
    movies : movies dataframe

    Returns
    -------
    movie_arr : preprocessed reviews list
    ratings_arr : preprocessed ratings list
    d : movie-cat_id to movie-id dict
    movies : processed movie dataframe
    '''
    data['movieId'] = data['movieId'].astype('category')
    d = dict(enumerate(data['movieId'].cat.categories))
    data['movieId'] = data['movieId'].cat.codes
    movies = movies.set_index('movieId')
    num_movies = len(movies)
    print("Number of movies : ",num_movies)
    movie_arr,ratings_arr = convert(data, num_movies)
    
    return movie_arr, ratings_arr, d, movies
    

if __name__ == '__main__':
    data, movies = load_dataset()
    pre_process(data, movies)
    
