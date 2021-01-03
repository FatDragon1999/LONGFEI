#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation


# In[2]:


data = pd.read_csv('./ml-100k/u.data')
data.head()


# In[3]:


data = pd.read_csv('./ml-100k/u.data',sep = '\t', names = ['user_id','item_id','rating','timstamp'])


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data,test_size = 0.4) 
# 训练集与测试集70:30比例拆分


# In[8]:


train_data.describe()


# In[9]:


# 此时的data就为用户评分表，即复现中的movie_rating
# 创建用户-电影评分表
rating_matrix = train_data.pivot_table(index = ['item_id'],columns = ['user_id'],
                                values = 'rating').reset_index(drop = True)
rating_matrix.fillna(0, inplace = True)
user_item = rating_matrix
user_item.head()


# In[10]:


user_item.shape


# In[11]:


# 训练集构造见物品相似矩阵
movie_similarity = 1 - pairwise_distances(rating_matrix.values, metric = "cosine")
np.fill_diagonal(movie_similarity,0)
rating_matrix = pd.DataFrame(movie_similarity)
rating_matrix.head()


# In[12]:


rating_matrix.shape


# In[13]:


movie_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movie = pd.read_csv('./ml-100k/u.item', sep = '|', names = movie_cols, encoding = 'latin-1',usecols = ["movie_id","title","release_date","video_release_date","imdb_url"])
movie.drop(movie.columns[[3,4]], axis = 1, inplace =  True)
movie.head()


# In[14]:


# 训练集推荐TOP_N
user_inp = "Four Rooms (1995)" 
inp = movie[movie['title'] == user_inp].index.tolist()
inp = inp[0]
movie['similarity'] = rating_matrix.iloc[inp]
movie.columns=['movie_id', 'title', 'release_date', 'similarity']
movie.head(5)


# In[15]:


# 测试集构造
# 此时的data就为用户评分表，即复现中的movie_rating
# 创建用户-电影评分表
rating_test_matrix = test_data.pivot_table(index = ['item_id'],columns = ['user_id'],
                                values = 'rating').reset_index(drop = True)
rating_test_matrix.fillna(0, inplace = True)
user_item_test = rating_test_matrix
user_item_test.head()


# In[16]:


user_item_test.shape


# In[17]:


# 测试集构造见物品相似矩阵
movie_similarity_test = 1 - pairwise_distances(rating_test_matrix.values, metric = "cosine")
np.fill_diagonal(movie_similarity_test,0)
rating_matrix_test = pd.DataFrame(movie_similarity_test)
rating_matrix_test.head()


# In[18]:


rating_matrix_test.shape


# In[19]:


# 测试集top-N推荐
user_inp = "Four Rooms (1995)" 
inp = movie[movie['title'] == user_inp].index.tolist()
inp = inp[0]
movie['similarity'] = rating_matrix_test.iloc[inp]
movie.columns=['movie_id', 'title', 'release_date', 'similarity']
movie.head(5)


# In[20]:


# 训练集预测，求得RMSE
# 将dataframe形式的用户物品相似矩阵转为numpy 数组
rating_matrix_array = rating_matrix.values
user_item_pre = rating_matrix_array.dot(movie_similarity) / np.array([np.abs(movie_similarity).sum(axis = 1)])

pre_flatten = user_item_pre
rating_matrix_flatten = rating_matrix_array
from sklearn.metrics import mean_squared_error
from math import sqrt
error_train = sqrt(mean_squared_error(pre_flatten, rating_matrix_flatten))
print(error_train)


# In[21]:


# 测试集预测，求得RMSE
rating_test_matrix_array = rating_matrix_test.values
user_item_test_pre = rating_test_matrix_array.dot(movie_similarity_test) / np.array([np.abs(movie_similarity_test).sum(axis = 1)])

test_pre_flatten = user_item_test_pre
test_rating_matrix_flatten = rating_test_matrix_array
error_test = sqrt(mean_squared_error(test_pre_flatten, test_rating_matrix_flatten))


# In[22]:


print(error_test)


# In[ ]:




