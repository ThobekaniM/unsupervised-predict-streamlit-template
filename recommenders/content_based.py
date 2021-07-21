"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
from operator import index
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


# Importing data
# df_sample_submission = pd.read_csv('sample_submission.csv')
df_movies = pd.read_csv('resources/data/movies.csv')
df_imdb = pd.read_csv('resources/data/imdb_data.csv')
df_genome_scores = pd.read_csv('resources/data/genome_scores.csv')
df_genome_tags = pd.read_csv('resources/data/genome_tags.csv')
df_train = pd.read_csv('resources/data/train.csv')
df_test = pd.read_csv('resources/data/test.csv')
df_tags = pd.read_csv('resources/data/tags.csv')
df_links = pd.read_csv('resources/data/links.csv')


# make a copy of the train dataset to work on
train_copy = df_train.copy()
# remove the timestamp column from the copy in order to be able to build models on train data that matches the test data
# we will evaluate the importance of the time stamp column later
train_copy = train_copy.drop('timestamp', axis = 1)

# function to preprocess data
# we use a subset of the data for computation purposes
def data_preprocessing (subset_size = 12000):
    """Prepare data for use within Content filtering algorithm.

    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.

    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.

    """

    
    # Inner join the imdb dataframe with the movies dataframe
    imdb = df_imdb[['movieId','title_cast','director', 'plot_keywords']]
    merge = imdb.merge(df_movies[['movieId', 'genres', 'title']], on='movieId', how='inner')

    # Convert data types to string in order to do string manipulation
    merge['title_cast'] = merge.title_cast.astype(str)
    merge['plot_keywords'] = merge.plot_keywords.astype(str)
    merge['genres'] = merge.genres.astype(str)
    merge['director'] = merge.director.astype(str)

    # clean directors and title_cast column
    # remove spaces and "|"
    merge['director'] = merge['director'].apply(lambda x: "".join(x.lower() for x in x.split()))
    merge['title_cast'] = merge['title_cast'].apply(lambda x: "".join(x.lower() for x in x.split()))
    merge['title_cast'] = merge['title_cast'].map(lambda x: x.split('|'))
    #convert title cast back to string and remove commas
    merge['title_cast'] = merge['title_cast'].apply(lambda x: ','.join(map(str, x)))
    merge['title_cast'] = merge['title_cast'].replace(',',' ', regex=True)
    
    # clean plot keywords column
    # remove spaces and "|"
    merge['plot_keywords'] = merge['plot_keywords'].map(lambda x: x.split('|'))
    merge['plot_keywords'] = merge['plot_keywords'].apply(lambda x: " ".join(x))

    # clean plot genres column
    # remove spaces and "|" 
    merge['genres'] = merge['genres'].map(lambda x: x.lower().split('|'))
    merge['genres'] = merge['genres'].apply(lambda x: " ".join(x))

    
    #subset table to only return required columns
    df_features = merge[['title_cast','director','plot_keywords','genres']]

    #we combine the features columns into  single string
    merge['combined_features'] = df_features['title_cast'] +' '+ df_features['director'] +' '+ df_features['plot_keywords'] +' '+ df_features['genres']
    merge_subset = merge[:subset_size]
    
    return merge_subset


# Preprocess data
processed_df = data_preprocessing(12000)
print(processed_df.columns)
#define the count vectorizer
cv = CountVectorizer()

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
#function to obtain recommendations
def content_model(titles,n):
    '''
    title: title that user will enter
    n: the number of recommendations required
    cv_matrix: unpickled countvecorizer
    
    '''
    df = processed_df.copy()
    cv_model = cv.fit_transform(df['combined_features'])
    #set title column as the index and create a dataframe of titles
    df.set_index('title', inplace = True)
    indices = pd.DataFrame(df.index)
    #create the cosine similarity matrix using the count vectorizer
    sim_score = cosine_similarity(cv_model,cv_model)
    
    #create an empty list of the recommended movies
    recommended_movies = []
    
    for title in titles:
        # match the entered title to its index in the titles dataframe
        idx = indices[indices == title].index[0]

        # get the similarity scores from highest to lowest  
        score_series = pd.Series(sim_score[idx]).sort_values(ascending = False)

        # create a list of the top nth title indexes from the similarity matrix
        top_n_indexes = list(score_series.iloc[1:int(n/3)].index)
    
        # add the titles that match the indexes to the list
        for i in top_n_indexes:
            recommended_movies.append(list(df.index)[i])
        
    return recommended_movies
