#!/usr/bin/env python
# coding: utf-8

# In[20]:


from modelarts.session import Session
session = Session()
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation
from sklearn.model_selection import train_test_split


# In[21]:


# 创建电影评分信息表
movie_rating_cols = ['user_id',  'movie_id', 'rating', 'unix_tiemstamp'] # set the table and col_name,define it use_rating_cols
movie_rating = pd.read_csv('./ml-100k/u.data',sep = '\t', names = movie_rating_cols, parse_dates = True)
movie_rating.shape


# In[22]:


# 创建电影信息表
movie_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movie = pd.read_csv('./ml-100k/u.item', sep = '|', names = movie_cols, encoding = 'latin-1',usecols = ["movie_id","title","release_date","video_release_date","imdb_url"])
movie.shape


# In[23]:


# 创建用户评分表
movie_rating_matrix = movie_rating.pivot_table(index = ['user_id'], columns = ['movie_id'],
                                                values = 'rating')
movie_rating_matrix.fillna(0, inplace = True)
cmu = movie_rating_matrix
cmu.head()


# In[24]:


# 创建共同评分向量
def build_gather_ratings(user_id1,user_id2):
    bool_array = cmu.loc[user_id1].notnull() & cmu.loc[user_id2].notnull()
    return cmu.loc[user_id1, bool_array], cmu.loc[user_id2, bool_array]


# In[25]:


# 皮尔逊相关系数
def pearson(user_id1, user_id2):
    x, y = build_gather_ratings(user_id1, user_id2)
    meanx, meany = x.mean(), y.mean()
    # 分母
    denominator = (sum((x-meanx) ** 2) * sum((y-meany) ** 2)) ** 0.5
    try:
        value = sum((x - meanx) * (y - meany)) / denominator
    except ZeroDivisionError:
        value = 0
    return value
 


# In[26]:


# 建立相似用户矩阵
def nearest_neighbor(user_id, k = 3):
    return cmu.drop(user_id).index.to_series().apply(pearson, args = (user_id,)).nlargest(k)
# 尝试
nearest_neighbor(1,)


# In[27]:


# 推荐最相似的用户ID
def recommend(user_id):
    nearest_user_id = nearest_neighbor(user_id).index[0] # 取返回的第一列最相似的用户id
    print('最相似的用户id：',nearest_user_id)
    # 返回电影id及项目评分
    return cmu.loc[nearest_user_id, cmu.loc[user_id].isnull() & cmu.loc[nearest_user_id].notnull()].sort_values()


# In[29]:


recommend(1)


# In[ ]:




