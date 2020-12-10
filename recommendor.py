import pandas as pd
import json
import ast
import numpy as np 
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

def retrieve_data():
    reviews_Musical_Instruments_5 = pd.read_json('data/reviews_Musical_Instruments_5.json',lines=True)
    meta_list = []
    with open('data/meta_Musical_Instruments.json') as f:
        for line in f.readlines():
            line = json.dumps(ast.literal_eval(line))
            d = json.loads(line)
            if 'title' not in d:
                continue
            else:
                meta = [d['asin'], d['title'], d['imUrl']]
                meta_list.append(meta)
    
    column_names = ['asin','title','image_url']
    meta = pd.DataFrame(meta_list, columns=column_names)
    # merge two datasets
    merged_df = reviews_Musical_Instruments_5[['asin','overall','unixReviewTime']].merge(meta[['asin', 'title','image_url']], on='asin', how='inner')
    # each row: asin, overall, unixReviewTime, title, image_url
    return merged_df

def popularity_recommend(merged_df, num):
    # get each product's latest released review time
    latest_time_df = merged_df[merged_df.groupby('asin')['unixReviewTime'].transform('max') == merged_df['unixReviewTime']]
    # get each product's mean rating
    filtered_df = merged_df.groupby(['asin','title','image_url'])['overall'].mean()
    filtered_df = filtered_df.reset_index(drop=False)
    # merge two df and sort by ratings then review time
    result = filtered_df[['asin','title','image_url','overall']].merge(latest_time_df[['asin','unixReviewTime']], on='asin', how='inner')
    result = result.sort_values(['overall', 'unixReviewTime'], ascending=[False, False])
    # each row: asin, title, image_url, overall, unixReviewTime, 
    return result.head(num)

def get_rating_matrix():
    df = pd.read_csv("data/ratings_Musical_Instruments.csv",names=['userId', 'productId','Rating','timestamp'])
    df.drop(['timestamp'], axis=1,inplace=True)
    # ###Check the distribution of the rating
    # with sns.axes_style('white'):
    #     g = sns.factorplot("Rating", data=df, aspect=2.0,kind='count')
    #     g.set_ylabels("Total number of ratings")

    #Getting the new dataframe which contains users who has given 50 or more ratings
    new_df=df.groupby("productId").filter(lambda x : x['Rating'].count() >= 50)
    ratings_matrix = new_df.pivot_table(values='Rating', index='userId', columns='productId', fill_value=0)   
    return ratings_matrix

def recommend(merged_df, p_id):
    ratings_matrix = get_rating_matrix()
    # Transpose the matrix
    X = ratings_matrix.T
    SVD = TruncatedSVD(n_components=10)
    decomposed_matrix = SVD.fit_transform(X)

    #Correlation Matrix: product to product correlation
    correlation_matrix = np.corrcoef(decomposed_matrix)
    # find the position of given product in the list of all products
    product_names = list(X.index)
    product_ID = product_names.index(p_id)
    correlation_product_ID = correlation_matrix[product_ID]
    recommend = list(X.index[correlation_product_ID > 0.65])
    # Removes the item already bought by the customer
    recommend.remove(p_id) 

    results = merged_df[merged_df.asin.isin(recommend)].groupby(['asin','title','image_url'])['overall'].mean()
    # results = results.sort_values('overall', ascending=False)
    # # asin, title, image_url, overall
    return results[:5]

if __name__ == "__main__":
    df = retrieve_data()
    print(recommend(df, 'B0002CZSJY'))