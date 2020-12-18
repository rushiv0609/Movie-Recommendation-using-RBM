# Movie-Recommendation-Engine-Using-RBM
A Movie Recommender System using Restricted Boltzmann Machine (RBM) approach used is collaborative filtering. This system is an algorithm that recommends items by trying to find users that are similar to each other based on their item ratings.  

RBM is a Generative model with two layers(Visible and Hidden) that assigns a probability to each possible binary state vectors over its visible units. Visible layer nodes have visible bias(vb) and Hideen layer nodes have hidden bias(hb). A weight matrix of row length equal to input nodes and column length equal to output nodes.

**If you have CUDA enabled GPU in PyTorch then it will be selected by defualt**

## Dataset

We can use any dataset variant from MovieLens. In load_data.py->load_dataset(size), there are 3 choices as following:  

1. small : [http://files.grouplens.org/datasets/movielens/ml-latest.zip](http://files.grouplens.org/datasets/movielens/ml-latest.zip)  
1. medium : [https://grouplens.org/datasets/movielens/10m/](https://grouplens.org/datasets/movielens/10m/)
1. large : [https://grouplens.org/datasets/movielens/25m/](https://grouplens.org/datasets/movielens/25m/)

## Preprocessing

Since the movie-ids are not continous i.e. there are some gaps in numbering, we encode all movie-ids to category-ids and save it to a dictionary.  

We create 2 lists : movies and ratings where each element in the list contains the following information : 

- movies-list : list of ids of movies a user has rated  
- ratings-list : list of ratings given by user given to movies corresponding to movies-list. Rating are normalized to be in range [0,1]


## Model
RBM has 2 layers : Visible and Hidden

- Visible layer size : Number of movies
- Hidden layer size : 1024 units

# Steps to run
First install the required packages
```
git clone https://github.com/rushiv0609/Movie-Recommendation-using-RBM.git
pip install -r requirements.txt
python main.py
```

## Training 
Each input batch is transformed into a torch tensor of shape [batch-size, number-of-movies] using movies-list & ratings-list. Then this transformed input is passed to RBM.

k-contrastive divergence is used to train the weights and bias of RBM.

## Recommending

We can use recommend() function in recommend.py to recommend top 10 movies by passing user-id as parameter
