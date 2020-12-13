# SI650
SI650 Final Project:

##Amazon product recommendation system based on reviewer’s review history

This project is composed of two parts: recommender and search engine.
Dataset are retrieved from: http://jmcauley.ucsd.edu/data/amazon/links.html

Instructions on codes: 

1. Recommendor_final.ipynb: 
	Recommender code is wrapped in this file.
	a. popularity_recommend(merged_df, num): 
		In order to recommend newest, popular and highest rated products. Simply sort all retrieved unique product id by most rated, normalized rating of all reviews, most up-to date review time. 
			--merged_df: retrieved dataframe
			--num: recommended items number 
	b.recommend(rating_file_name, p_id, correlation): 
		Based on machine learning, a model is trained to make predictions about items with highest possibilities that a user choosing a product might also like. 
			--rating_file_name: target rating file
			--p_id: target product id
			--correlation: threshold for filtering products in correlation matrix 
			
2.   METAPY_BM25.ipynb:
	BM25 score of each review after tokenizing and removing stop-words can be computed. And highest reviews’ of 10 products will be retrieved as most relevant products with input query given by user.
	a. get_dat_file(df):
		all reviews in the dataframe are retrieved and put into review_data.dat file
			--df: data frame with all docs/reviews
  b. review_data/test-queries.txt: contains 20 test queries which can be customized
	c. Polarity: Exhibits polarity distribution of reviews in given dataset
	
	
3.  SI650_final_version.ipynb:
	This file contains our main method in Searching Engine.
	a. get_result(query, entire_merged_df, feedback): 
		retrieve top10 products with given query, and merged data frame retrieved from dataset and feedback cache
	b. update_cache(asin_rel_tuple, query, feedback):
		Update feedback cache with each tuple of (product_id, relevance) of query.
	c. review_data/test-queries.txt: contains 20 test queries which can be customized
	
	
4.  evaluation.ipynb: 
	After annotating results generated in the search engine, an annotation csv file can be imported into this file and be evaluated or analysed with matrices NDCG@10, precision@5, PR  Curve
	a. evaluate(annotation_df, k): 
		Return mean NDCG@10 and mean precision@k with the given annotated data frame containing relevance. And print PR Curve of each query’s 10 relevance.


5.  results:
	a. bm25-annotation-final.csv: 
	Results returned from METAPY_BM25.ipynb, and manually annotated in column ‘relevance’. Annotation file is imported into evaluation.ipynb to evaluate bm25 performance
	b. main-results-annotation.csv:
	Results returned from SI650_final_version.ipynb, and manually annotated in column ‘relevance’. Annotation file is imported into evaluation.ipynb to evaluate our method’s performance
