"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# from utils import movies
movies_df = pd.read_csv("./processed-data/movies_df_processed_2.csv", sep=",")
# 9742 rows × 610 columns
rating_matrix = pd.read_csv("./processed-data/rating_matrix_df.csv", sep=',',index_col=0)
userid_list = rating_matrix.columns.to_list()
movieid_list = movies_df['movieId'].to_list()
new_userId = max([int(id) for id in userid_list])+1

nmf_model = pickle.load(open('model_nmf.pkl', 'rb'))
Q_df = pd.DataFrame(nmf_model.components_, columns=movieid_list, index=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])


def recommend_random(k=3):
    return movies_df['title'].sample(k)

def recommend_with_cos_similarity(input_rating, k=3):
    '''
    pass input rating dict with movieId and rating
    return rec_titles
    '''
    input_full_rating = get_input_full_rating(input_rating)
    input_full_rating_df = pd.DataFrame(input_full_rating, index=[new_userId])
    df = pd.concat([rating_matrix, input_full_rating_df.transpose()], axis=1, join='outer')
    # print("df", df.shape) # 9742 rows × 611 columns

    unseen_rating_dic = get_unseen_rating_dic(df, new_userId, 50)
    rec_dic_desc = dict(sorted(unseen_rating_dic.items(), key=lambda x:x[1]))
    rec_list = list(rec_dic_desc.keys())
    rec_titles = get_rec_titles(movies_df, rec_list, k)
    return rec_titles

def recommend_with_nmf(input_rating, k=3):
    """
    NMF Recommender
    INPUT
    - user_vector with shape (1, #number of movies)
    - user_item_matrix
    - trained NMF model
    OUTPUT
    - a list of movieIds
    """
    input_full_rating = list(get_input_full_rating(input_rating).values())
    input_processed = np.where(np.isnan(input_full_rating), np.nanmean(input_full_rating), input_full_rating)
    input_processed = input_processed.reshape(1, -1)
    # print("input_processed", input_processed.shape) #(1, 9742)
    input_P = nmf_model.transform(input_processed)
    input_R = np.dot(input_P, Q_df)
    recommendation = pd.DataFrame({'user_input':input_full_rating,'predicted_ratings':input_R[0]}, index = movieid_list)
    rec_df = recommendation[recommendation['user_input'].isna()].sort_values(by = 'predicted_ratings', ascending= False)
    rec_list = rec_df.index.to_list()
    
    rec_titles = get_rec_titles(movies_df, rec_list, k)
    return rec_titles


def get_input_full_rating(input_rating):
    """
    transfer user_rating to user_full_rating
    with given value and nan for missing value
    INPUT
    - user_rating (3 items) dict with movieId and rating
    OUTPUT
    -user_full_rating dict with movieId and rating
    """
    movieId = movies_df['movieId'].to_list()
    input_full_rating = dict(zip(movieId, [np.nan]*len(movieId)))
    for key, value in input_rating.items():
       input_full_rating[key] = value
    return input_full_rating

# for cosine similarity
def get_cos_df(df):
    '''
    pass rating_matrix and return cosine dataframe
    '''
    df = df.fillna(value = 0)
    cs = cosine_similarity(df.T)
    cs_df = pd.DataFrame(cs, columns = df.columns, index = df.columns).round(2)
    # print("cs_df", cs_df.shape) #611 rows × 611 columns
    return cs_df

def get_unseen_rating_dic(df, userId, n_similarity):
    '''
    '''
    cs_df = get_cos_df(df)
    unseen = df[df[userId].isna()].index
    similar_users = cs_df[userId].sort_values(ascending= False).index[1:n_similarity+1]
    result_dic = {}
    
    for movie in unseen:
        other_users = df.columns[~df.loc[movie].isna()] 
        other_users = set(other_users)
        num = 0
        den = 0
        overlapped_users = set(similar_users).intersection(other_users)
        if len(overlapped_users) != 0:
            for user in overlapped_users: 
                rating = df[user][movie]     # extract relevant ratings from user_item
                sim = cs_df[userId][user]   # extract relevant cosine sim values 
                num = num + (rating*sim)            # account for "level of similarity"
                den = den + sim + 0.000001
            pred_ratings = num/den
            # print(movie, pred_ratings)
            result_dic.update({movie: pred_ratings})
        else:
            # print(movie, pred_ratings)
            pred_ratings = 2.5
            result_dic.update({movie: pred_ratings})
    return result_dic


def get_rec_titles(movies_df, rec_list, n_reco):
    rec_list = rec_list[:n_reco]
    rec_movie_titles = []
    for movie_id in rec_list:
        row = movies_df[movies_df['movieId'] == int(movie_id)]
        title = row["title"].iloc[0]
        rec_movie_titles.append(title)
    
    return rec_movie_titles

# def recommend_neighborhood(query, model, k=3):
#     """
#     Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model. 
#     Returns a list of k movie ids.
#     """   
#     pass
    

