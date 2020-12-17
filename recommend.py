import pickle
import numpy as np
import pandas as pd
import torch
from RBM import RBM

def recommend(rbm, user_id, data, num_movies):
    '''
    Parameters
    ----------
    user_id : id of user in dataframe

    Returns
    -------
    movie-cat_ids of top 10 recommended movies

    '''
    # convert user data to RBM Input
    device = rbm.W.device
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



