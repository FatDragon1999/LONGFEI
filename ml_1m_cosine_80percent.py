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


# 拆分训练集与数据集
from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(movie_rating,test_size = 0.2)


# In[10]:


# 创建用户电影矩阵
movie_rating_matrix = train_data.pivot_table(index = ['movie_id'], columns = ['user_id'],
                                              values = 'rating').reset_index(drop = True)
movie_rating_matrix.fillna(0, inplace = True)
cmu = movie_rating_matrix
cmu.head()


# In[11]:


# 利用余弦算法获得电影相似度矩阵
movie_similarity = 1 - pairwise_distances(movie_rating_matrix.values, metric = "cosine")
np.fill_diagonal(movie_similarity,0)
# 将相似度填回相似矩阵
movie_rating_matrix = pd.DataFrame(movie_similarity)
movie_rating_matrix.head()


# In[12]:


# 获得TOP-N推荐
user_inp = "Toy Story (1995)"
inp = movie[movie['Title'] == user_inp].index.tolist()
inp = inp[0]
movie['similarity'] = movie_rating_matrix.iloc[inp]
movie.columns = ['MovieID','Title','Genres','similarity']
movie.head()


# In[13]:


# 测试集合构造
rating_test_matrix = test_data.pivot_table(index = ['movie_id'],columns = ['user_id'],
                                values = 'rating').reset_index(drop = True)
rating_test_matrix.fillna(0,inplace = True)
cmu_test = rating_test_matrix
cmu_test.head()


# In[14]:


# 测试集构造物品相似矩阵
movie_similarity_test = 1 - pairwise_distances(rating_test_matrix.values, metric = "cosine")
np.fill_diagonal(movie_similarity_test,0)
rating_matrix_test = pd.DataFrame(movie_similarity_test)
rating_matrix_test.head()


# In[15]:


# 测试集TOP-N推荐
user_inp = "Toy Story (1995)"
inp = movie[movie['Title'] == user_inp].index.tolist()
inp = inp[0]
movie['similarity'] = rating_matrix_test.iloc[inp]
movie.columns = ['MovieID','Title','Genres','similarity']
movie.head()


# In[16]:


# 训练集预测，求得RMSE
# 将dataframe形式的用户物品相似矩阵转为numpy 数组
rating_matrix_array = movie_rating_matrix.values
user_item_pre = rating_matrix_array.dot(movie_similarity) / np.array([np.abs(movie_similarity).sum(axis = 1)])

pre_flatten = user_item_pre
rating_matrix_flatten = rating_matrix_array
from sklearn.metrics import mean_squared_error
from math import sqrt
error_train = sqrt(mean_squared_error(pre_flatten, rating_matrix_flatten))
print(error_train)


# In[17]:


# 测试集预测，求得RMSE
rating_test_matrix_array = rating_matrix_test.values
user_item_test_pre = rating_test_matrix_array.dot(movie_similarity_test) / np.array([np.abs(movie_similarity_test).sum(axis = 1)])

test_pre_flatten = user_item_test_pre
test_rating_matrix_flatten = rating_test_matrix_array
error_test = sqrt(mean_squared_error(test_pre_flatten, test_rating_matrix_flatten))
print(error_test)


# In[ ]:




