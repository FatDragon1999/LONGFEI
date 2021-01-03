#!/usr/bin/env python
# coding: utf-8

# In[1]:


from modelarts.session import Session
session = Session()
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation
from sklearn.model_selection import train_test_split


# In[2]:


# 创建用户信息表
users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('./ml-100k/u.user', sep='|', names=users_cols, parse_dates=True)
users.shape


# In[3]:


# 创建电影评分信息表
movie_rating_cols = ['user_id',  'movie_id', 'rating', 'unix_tiemstamp'] # set the table and col_name,define it use_rating_cols
movie_rating = pd.read_csv('./ml-100k/u.data',sep = '\t', names = movie_rating_cols, parse_dates = True)
movie_rating.shape


# In[4]:


# 创建电影信息表
movie_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movie = pd.read_csv('./ml-100k/u.item', sep = '|', names = movie_cols, encoding = 'latin-1',usecols = ["movie_id","title","release_date","video_release_date","imdb_url"])
movie.shape


# In[5]:


# 合并电影和电影评分信息
movie_ratings = pd.merge(movie, movie_rating)
# 在movie_ratings的基础上合并用户信息
dataframe = pd.merge(movie_ratings, users)
dataframe.shape


# In[6]:


# 清除无效信息
dataframe.drop(dataframe.columns[[3,4,7]], axis = 1, inplace = True)
movie_rating.drop("unix_tiemstamp", inplace = True, axis = 1)
movie.drop(movie.columns[[3,4]], axis = 1, inplace =  True)
dataframe.head()


# In[7]:


# 创建用户评分表
movie_rating_matrix = movie_rating.pivot_table(index = ['movie_id'], columns = ['user_id'],
                                                values = 'rating').reset_index(drop = True)
movie_rating_matrix.fillna(0, inplace = True)
cmu = movie_rating_matrix
cmu.head()


# In[8]:


movie_similarity = pairwise_distances(movie_rating_matrix.values, metric = "euclidean")
np.fill_diagonal(movie_similarity,0)
movie_rating_matrix = pd.DataFrame(movie_similarity)
# 归一化
movie_rating_matrix_sim = ( movie_rating_matrix -  movie_rating_matrix.min()) / ( movie_rating_matrix.max() -  movie_rating_matrix.min())
movie_rating_matrix_sim.head()
# 使用 movie_similarity = 1 / pairwise_distances(movie_rating_matrix.values, metric = "euclidean")以归一化时
# 因为有的矩阵值为0，因此出现了1/0的错误，利用函数进行转换时则在矩阵行列重新赋值上出现错误，需要重新改进，于是使用pandas中的归一方法 


# In[9]:


movie_rating_matrix_sim.shape


# In[12]:


# 推荐相似性较高的前5部
user_inp = "Copycat (1995)" # have a blank between cat and (), attention!
inp = movie[movie['title'] == user_inp].index.tolist()
# movie['title'] == user_inp条件
# movie[movie['title'] == user_inp] 条件所在行
# movie[movie['title'] == user_inp].index.tolist() 确定这些行所在列
inp = inp[0]
movie['similarity'] = movie_rating_matrix_sim.iloc[inp]
# iloc 基于索引确定
movie.columns=['movie_id', 'title', 'release_date', 'similarity']
movie.head(5)


# In[ ]:




