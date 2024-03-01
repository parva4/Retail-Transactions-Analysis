import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
import os
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from collections import Counter
warnings.filterwarnings("ignore")

path = '/Users/parva4/Documents/retTransac'
df = pd.read_csv(os.path.join(path, 'cleanDf.csv'))

'''defectTags = defective = ['Unsaleable, destroyed.', 'Damaged', 'damages', 'mouldy, unsaleable.', 'wet boxes', 'wet', 'wet rusty']
df = df[~(df['Description'].isin(defectTags))]
lostTags = [prod for prod in set(df['Description']) if ('lost' in prod) or  ('mixed' in prod) or ('?' in prod)]
df = df[~(df['Description'].isin(lostTags))]'''

products_dict = pickle.load(open('products.pk', 'rb'))
products = [prod[0] for prod in products_dict.values()]

#creating bag of words for product names to find cosine similarity
ctVectorizer = CountVectorizer()
bow = ctVectorizer.fit_transform(products)

#finding cosine similarity in order to later perform k-means clustering on it
'''similarity_matrix = cosine_similarity(bow, bow)

#using elbow method to find optimal value of k
k_values = range(2, 50)
inertia = []
for k in tqdm(k_values):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(similarity_matrix)
    inertia.append(kmeans.inertia_)

delta = np.diff(inertia)
kOpt = np.argmax(delta)

kmeans = KMeans(n_clusters=kOpt, random_state=42)
kmeans.fit(similarity_matrix)

f = open('kmeans_retail.pk', 'wb')
pickle.dump(kmeans, f)'''

#loading the saved kmeans model
kmeans_model = pickle.load(open('kmeans_retail.pk', 'rb'))
labels = kmeans_model.labels_
prodgrp = {}

#creating a dictinoary to store group number corresponding to each stockcode
for i, prod in enumerate(products_dict):
    prodgrp[prod] = labels[i]

#appending groupnumber to the og dataframe
df['ProdGrp'] = df['StockCode'].map(prodgrp)
grpSum = df.groupby('ProdGrp')['TotalCost'].sum()
revenueContri = pd.DataFrame(round(df.groupby(['ProdGrp', 'Description'])['TotalCost'].sum().groupby('ProdGrp').nlargest(5) * 100 / grpSum, 2))
revenueContri.reset_index(level=[1, 'Description'], inplace=True)
revenueContri.reset_index(drop=True, inplace=True)
revenueContri.rename(columns={'TotalCost' : '%Revenue'}, inplace=True)

#plot to find the top 5 products for each cluster that contribute the most to the overall cluster revenue
def topRevProductsCluster(revenueContridf):
    fig = go.Figure()
    for grp in revenueContridf['ProdGrp'].unique():
        temp = revenueContridf[revenueContridf['ProdGrp'] == grp]
        fig.add_trace(go.Bar(x=temp['Description'], y=temp['%Revenue'], name=str(grp)))

    steps = []
    for i, grp in enumerate(revenueContridf['ProdGrp'].unique()):
        step = dict(
            method="update",
            label=str(grp),
            args=[{"visible": [str(grp) == trace.name for trace in fig.data]}],
        )
        steps.append(step)

    sliders = [dict(active=0, currentvalue={'prefix' : 'Cluster: '}, pad={"t": 50},
                steps=steps)]

    fig.update_layout(
        sliders=sliders,
        title="Top 5 Renenue Contributing Products to each Cluster",
        xaxis_title="Product Description",
        yaxis_title="%Revenue Contribution to the Group"
    )

    fig.show()
    fig.write_html('Top5ProdsPerCluster.html')
    return fig

#code below builds a plot that categorizes each group by country or revenue contribution by top 3 countreis
def topCountriesCluster(df):
    country_group_contri = df.groupby('Country')['TotalCost'].sum()
    grpSum = df.groupby('ProdGrp')['TotalCost'].sum()
    revenuePercent = pd.DataFrame(round(df.groupby(['ProdGrp', 'Country'])['TotalCost'].sum().groupby('ProdGrp').nlargest(3) * 100 / grpSum, 2))
    
    revenuePercent.reset_index(level=[1, 'Country'], inplace=True)
    revenuePercent.reset_index(drop=True, inplace=True)
    revenuePercent.rename(columns={'TotalCost': '%Revenue'}, inplace=True)
    revenuePercent['Log of %Revenue'] = revenuePercent['%Revenue'].apply(lambda x: math.log(x))
    
    x = revenuePercent['ProdGrp'].unique()
    countries = revenuePercent['Country'].unique()
    
    traces = []
    for country in countries:
        temp = revenuePercent[revenuePercent['Country'] == country]
        trace = go.Bar(x=temp['ProdGrp'], y=temp['Log of %Revenue'], name=country)
        traces.append(trace)
            
    fig = go.Figure(data=traces)
    fig.update_layout(barmode='stack', xaxis_title='Product Group', yaxis_title='Log(% Revenue)',
    title='Top 3 Countries Contributing to the Revenue of each Group')
    fig.write_html('GrpTop3RevCountry.html')
    fig.show()

topRevProductsCluster(revenueContri)
topCountriesCluster(df)
