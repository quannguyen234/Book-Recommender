from book_functions import *

import re
import os
import time
import random
import requests
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import system
from math import floor
from copy import deepcopy
from rake_nltk import Rake


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# Most simple rec system

def simple_rec(genre, length, popularity, dict, df):
    ''' 
    use genre_id_dict and df_simple
    '''
    poss_books = dict[genre]
    df = df.loc[poss_books]
    
    return filter_df(length, popularity, df)
    



# Rec system using only description

def get_recommendations(title, dff):
    '''
    Takes in a title and dataframe (use dff), then makes an abbreviated df containing only the titles and index number. 
    Returns top 10 similar books based on cosine similarity of vectorized description ALONE.
    '''
    title = find_title(title, dff)

    # tfidf vectorize descriptions in dff
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(dff['description'])

    # get dot product of tfidf matrix on the transposition of itself to get the cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # create a new dataframe with titles as index, and index as a feature
    indices = pd.Series(list(range(len(dff))), index=dff.index)
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]

    return dff.loc[indices[movie_indices].index][:11]




# helper functions for interfacing with the final recommendation system

def fail_to_find(df):
    final = input("That title did not match any of our books! Please try again, or enter 'quit!' to stop playing.")
    if final == 'quit!':
        return 0
    else:
        return find_title(final, df)
        
def find_title(guess, df):
    guess = guess.lower()
    final = []
    titles_list = {x.lower(): x for x in df.index}
    for possible in list(titles_list.keys()):
        if guess in possible:
               final.append(possible)
    if len(final) == 0:
        return fail_to_find(df)
    if len(final) == 1:
        print (f"\n Great! Looking for recomendations for the book: {titles_list[final[0]]}")
        return titles_list[final[0]]
    elif len(final) > 1:
        maybe = input(f"We found {len(final)} books that matched your search! Would you like to look thru them? If so enter'yes', otherwise enter 'no'.")
        if maybe == 'yes':
            print ("Is your book in this list? \n")
            maybe = input(f"{final}\n")
        for poss in final:
            end = input(f"Is your book {titles_list[poss]}? If so enter 'yes' and if not enter 'no'.")
            if end == 'yes':
                print (f"\n Great! Looking for recomendations for the book: {titles_list[poss]}")
                return titles_list[poss]
        return fail_to_find(df)
                     
                      
                      
                      
# Filter helper functions
                      
def return_pop_df(popularity, df):
    '''
    returns population filtered dataframe
    '''
    if popularity == 'deep cut':
        return df[df['num_ratings'] < 27000]
    if popularity == 'well known':
        return df[(df['num_ratings'] < 80000) & (df['num_ratings'] > 27000)]
    if popularity == 'super popular':
        return df[df['num_ratings'] > 80000]
    
def filter_df(length, popularity, df):
    if length != None:
        if length == 'long':
            df = df[(df['pages'] >= 350)]
        elif length == 'short':
            df = df[(df['pages'] < 350)]
        
    if popularity != None:
        df = return_pop_df(popularity, df)
    
    if len(df) > 10:
        return df[:11]
    else:
        return df


def return_pop_df(popularity, df):
    '''
    Returns dataframe with only the designated range of number of ratings 
    (helper function for filter)
    '''
    if popularity == 'deep cut':
        return df[df['num_ratings'] < 27000]
    if popularity == 'well known':
        return df[(df['num_ratings'] < 80000) & (df['num_ratings'] > 27000)]
    if popularity == 'super popular':
        return df[df['num_ratings'] > 80000]                      
                          
                                          
                      
# Final recommendation system WITHOUT filter                    
                      
def recommendations(title, df, sim, list_length=11, suppress=False):
    '''
    Return recommendations based on a count vectorized BoW comprised of book author, genres and description.
    Takes in title, list length, a dataframe, a similarity matrix and an option to suppress output.
    '''
    
    recommended_books = []
    
    title = find_title(title, df)
    if title == 0:
        print ('Try again later')
        return 0

    # creating a Series for the movie titles so they are associated to an ordered numerical list
    indices = pd.Series(list(range(len(df))), index=df.index)
    
    # getting the index of the book that matches the title
    idx = indices[title]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:list_length+1].index)
    
    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_books.append(list(df.index)[i])
    
    if suppress == False:
        print (f"\n We recommend \n {recommended_books}")

    return df.loc[recommended_books]    
                      
                      
      
# Final Rec system WITH filter
                      
def rec_w_filter(title, df_filter, sim, length=None, popularity=None): 

    title = find_title(title, df_filter)
    
    indices = pd.Series(list(range(len(df_filter))), index=df_filter.index)
    idx = indices[title]
    sim_scores = list(enumerate(sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:100] # because the book selected willl be most similar
    movie_indices = [i[0] for i in sim_scores]
    df = df_filter.iloc[movie_indices]
     
    return filter_df(length, popularity, df)                      

                      
