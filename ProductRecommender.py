import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from collections import Counter
warnings.filterwarnings("ignore")

#i have tried building a product recommender using cosine similarity to find similarities in purchasing habits for each customer
#cosine similarity was calculated after building a sparse matrix for each that contains 1 if a particular customer purchased-
#-a particular product, else 0

class CustomerProductRecommender:
    def __init__(self, path='/Users/parva4/Documents/retTransac'):
        self.path = path
        self.df = pd.read_csv(os.path.join(self.path, 'cleanDf.csv'))
        self.removeReturned()

    def removeReturned(self):
        defectTags = ['Unsaleable, destroyed.', 'Damaged', 'damages', 'mouldy, unsaleable.', 'wet boxes', 'wet', 'wet rusty']
        self.df = self.df[~(self.df['Description'].isin(defectTags))]
        lostTags = [prod for prod in set(self.df['Description']) if ('lost' in prod) or  ('mixed' in prod) or ('?' in prod)]
        self.df = self.df[~(self.df['Description'].isin(lostTags))]
        self.df = self.df[self.df['Quantity'] > 0]

    def vectorizePurchaseData(self):
        products = {prod : i for i, prod in enumerate(self.df['Description'].unique())}
        customers = list(self.df['CustomerID'].unique())
        purchases_vector = {cust_id : [0]*len(products) for cust_id in customers}

        for customer_id in customers:
            temp = self.df[self.df['CustomerID'] == customer_id]
            for product in temp['Description'].unique():
                purchases_vector[customer_id][products[product]] = 1

        purchases_df = pd.DataFrame(purchases_vector).T
        purchases_df.columns = products.keys()
        purchases_df.reset_index(inplace=True)
        purchases_df.rename(columns={'index' : 'CustomerID'}, inplace=True)
        return purchases_df

    def getNearestCustomers(self):
        vectorDf = self.vectorizePurchaseData()
        cosine_matrix = cosine_similarity(vectorDf.values[:, 1:])
        top_5_nearest_customers = []
        for i in range(cosine_matrix.shape[0]):
            top_5_indices = cosine_matrix[i].argsort()[-6:-1][::-1]
            top_5_nearest_customers.append(top_5_indices)

        nearest_customers = {cust: top_5_nearest_customers[i] for i, cust in enumerate(vectorDf['CustomerID'])}
        return nearest_customers

    def getRecommendProds(self, customerID):
        nearest_cust_dict = self.getNearestCustomers()
        prods_op_bought = set(self.df[self.df['CustomerID'] == customerID]['Description'].unique())
        nearest_customers = nearest_cust_dict[customerID]
        return set(self.df.loc[nearest_customers]['Description']) - prods_op_bought

rec = CustomerProductRecommender()
print(rec.getRecommendProds(13047)) #just pass the customerId here for the customer you want to see the recommendations for