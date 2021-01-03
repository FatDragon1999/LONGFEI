#!/usr/bin/env python
# coding: utf-8

# In[1]:


BUCKET_NAME='test-xwh01'
OBS_BASE_PATH=BUCKET_NAME

from modelarts.session import Session
session = Session()

session.download_data(bucket_path="test-xwh01/test-machine/ratings.dat", path="./ratings.dat")
session.download_data(bucket_path="test-xwh01/test-machine/users.dat", path="./users.dat")
session.download_data(bucket_path="test-xwh01/test-machine/movies.dat", path="./movies.dat")
# import same usefull libraries
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation
movie_rating = pd.read_csv('./ratings.dat')


# In[2]:


movie_rating_cols = ['user_id',  'movie_id', 'rating', 'unix_tiemstamp'] 
movie_rating = pd.read_csv('./ratings.dat',sep = '::', names = movie_rating_cols, parse_dates = True)
movie_rating.head()


# In[3]:


movie_rating.nunique()


# In[4]:


user=pd.read_csv('./users.dat',sep='::',names=['UserID','Gender','Age','Occupation','Zip-code'])
user.head()


# In[5]:


user.shape


# In[6]:


movie=pd.read_csv('./movies.dat',sep='::',names=['MovieID','Title','Genres'])
movie.head()


# In[7]:


movie.shape


# In[8]:


movie_rating.drop("unix_tiemstamp",inplace = True, axis = 1)
movie_rating.head()


# In[9]:


# 创建用户电影矩阵
movie_rating_matrix = movie_rating.pivot_table(index = ['movie_id'], columns = ['user_id'],
                                              values = 'rating').reset_index(drop = True)
movie_rating_matrix.fillna(0, inplace = True)
cmu = movie_rating_matrix
cmu.head()


# In[10]:


# 利用余弦算法获得电影相似度矩阵
movie_similarity = 1 - pairwise_distances(movie_rating_matrix.values, metric = "cosine")
np.fill_diagonal(movie_similarity,0)
# 将相似度填回相似矩阵
movie_rating_matrix = pd.DataFrame(movie_similarity)
movie_rating_matrix.head()


# In[11]:


# 获得TOP-N推荐
user_inp = "Toy Story (1995)"
inp = movie[movie['Title'] == user_inp].index.tolist()
inp = inp[0]
movie['similarity'] = movie_rating_matrix.iloc[inp]
movie.columns = ['MovieID','Title','Genres','similarity']
movie.head()


# In[ ]:




