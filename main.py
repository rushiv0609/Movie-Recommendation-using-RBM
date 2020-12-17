import pickle
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import time
from math import ceil
from load_data import *
from RBM import RBM

def recommend(user_id):
    '''
    Parameters
    ----------
    user_id : id of user in dataframe

    Returns
    -------
    movie-cat_ids of top 10 recommended movies

    '''
    # convert user data to RBM Input
    user_df = data[data['userId'] == user_id].values
    input = torch.zeros(num_movies)
    for row in user_df:
        input[int(row[1])] = row[2]/5
    input = input.unsqueeze(dim = 0).to(device)

    # Give input to RBM
    h, _h = rbm.calc_hidden(input)
    v, _ = rbm.calc_visible(_h)
    out = v.cpu().squeeze() # visible layer probabilities after 1 cycle

    input = input.squeeze()
    out[input > 0] = -1 # set the value of already rated movies by user to -1
    order = out.argsort(descending= True)[:10] # select 10 max values from the output vector which will be recommended
    return order # Return the movie-ids of top 10 recommended movies

data, movies = load_dataset()
movie_arr, ratings_arr, d, movies = pre_process(data, movies)
num_movies = len(movies)
# train-test split 80:20
movie_train, movie_test, ratings_train, ratings_test = train_test_split(movie_arr, ratings_arr, test_size = 0.2, shuffle = True)

print("Number of total users : ",movie_arr.shape[0])
print("Number of users for training : ",movie_train.shape[0])
print("Number of users for testing : ",movie_test.shape[0])

# Empty CUDA cache and create RBM object on GPU
torch.cuda.empty_cache()
n_visible = num_movies
n_hidden = 1024
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
rbm = RBM(n_visible, n_hidden, device, lr = 0.001)
print("Number of visible nodes = ", n_visible)
print("Number of hidden nodes = ", n_hidden)
print("------------------------------------------------------")

'''
Training Phase 
'''
print("\nTraining Started \n")
start_time = time.time()
epochs = 10
batch_size = 128
N = movie_train.shape[0]
num_batches = ceil(N / batch_size)
loss_fn = nn.MSELoss()
for epoch in range(epochs):
    loss = 0.0
    i = 0
    while i < N:
        m_data = movie_train[i : i + batch_size]
        r_data = ratings_train[i : i + batch_size]
        i += batch_size
        input = np.zeros((m_data.shape[0], num_movies))
        for ind in range(m_data.shape[0]):
            #print(len(m_data[ind]), len(r_data[ind]))
            input[ind, m_data[ind]] = r_data[ind]
        input = torch.Tensor(input)
        out = rbm.cont_div(input)
        out = out.to(device)
        input = input.to(device)
        loss += loss_fn(input, out).item()
    
    print("Epoch %s => Loss = %s"%(epoch+1, loss/num_batches))

print("Training time = %s"%(time.time()-start_time))
print("------------------------------------------------------")
'''
Testing Phase
'''

print("\nTesting Started")
batch_size = 128
N = movie_test.shape[0]
num_batches = ceil(N / batch_size)
loss_fn = nn.MSELoss()
loss = 0.0
i = 0
while i < N:
    m_data = movie_test[i : i + batch_size]
    r_data = ratings_test[i : i + batch_size]
    i += batch_size
    input = np.zeros((m_data.shape[0], num_movies))
    for ind in range(m_data.shape[0]):
        #print(len(m_data[ind]), len(r_data[ind]))
        input[ind, m_data[ind]] = r_data[ind]
    input = torch.Tensor(input)
    out = rbm.infer(input)
    out = out.to(device)
    input = input.to(device)
    loss += loss_fn(input, out).item()

print("\nTest Loss = %s\n"%(loss/num_batches))
print("------------------------------------------------------")

'''
Save Model and recommend movies
'''
path = 'RBM.pkl'
with open(path, 'wb') as output:
    pickle.dump(rbm, output)
print("\nRBM pickle file saved at : %s\n"%(os.path.abspath(path)))
print("------------------------------------------------------")
user_id = 6
inds = recommend(user_id)
rec_movies = [movies.loc[d[x.item()]]['title'] for x in inds]
print("\nRecommended Movies for User-id %s : \n"%(user_id))
for movie in rec_movies:
	print(movie)
