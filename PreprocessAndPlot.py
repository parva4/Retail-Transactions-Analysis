import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from collections import Counter
from tqdm import tqdm
import pickle
import os
import ast
go.layout.template = 'plotly'

path = '/Users/parva4/Documents/retTransac'
os.chdir(path)
df = pd.read_csv(os.path.join(path, 'Online Retail.csv'), na_values=['?', 'nan'])

class RetailTransactionsCln:
    def __init__(self, delOutliers=True):
        self.df = pd.read_csv('Online Retail.csv')
        self.delOutliers = delOutliers
        self.execute()
    
    def preprocess(self):
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        #self.df['Description'] = self.df['Description'].astype(str)
        self.df = self.df[~(self.df['Country'] == 'Unspecified')]
        self.df.reset_index(inplace=True)

    def fill_nulls(self):
        products = pickle.load(open('products.pk', 'rb'))
        self.df.loc[self.df['UnitPrice'] <= 0, 'UnitPrice'] = np.nan
        #self.df.loc[self.df['Quantity'] <= 0, 'Quantity'] = np.nan
        
        #print(self.df.isnull().sum()) #Description    1454, UnitPrice      2517

        descriptions, unitprices = self.df['Description'].tolist(), self.df['UnitPrice'].tolist()
        for i in range(len(self.df)):
            if pd.isna(descriptions[i]) and self.df.loc[i, 'StockCode'] in products:
                descriptions[i] = products[self.df.loc[i,'StockCode']][0]
                
            if (pd.isna(self.df.loc[i, 'UnitPrice']) or self.df.loc[i, 'UnitPrice'] <= 0) and self.df.loc[i, 'StockCode'] in products:
                unitprices[i] = products[self.df.loc[i, 'StockCode']][1]

        self.df['Description'] = descriptions
        self.df['UnitPrice'] = unitprices

        #print(self.df.isnull().sum()) #Description  113, UnitPrice  134
        #self.avg_qty_customer_prod()
        self.df.dropna(inplace=True)
        self.df['TotalCost'] = round(self.df['Quantity'] * self.df['UnitPrice'], 2)
    
    def getOutliers(self, variable):
        q1 = np.quantile(self.df[variable], 25)
        q3 = np.quantile(self.df[variable], 75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        return lower, upper

    def avg_qty_customer_prod(self):
        self.df['Quantity'] = round(self.df.groupby(['CustomerID', 'StockCode'])['Quantity'].transform(lambda x: np.nanmean(x)))
        
    def execute(self):
        self.preprocess()
        self.fill_nulls()
        self.df.drop(columns=['index'], inplace=True)

'''rtt = RetailTransactionsCln()
new_df = rtt.df
new_df.to_csv(os.path.join(path, 'cleanDf.csv'))'''

df1 = pd.read_csv(os.path.join(path, 'cleanDf.csv'))

class MakePlots:
    def __init__(self, cleanDf_path):
        self.df1 = pd.read_csv(cleanDf_path)
        self.sold = self.df1[self.df1['Quantity'] >= 0]

    def topProductsPerCountryPlt(self): #top products according to country plot
        top_prods = self.df1.groupby(['Country', 'Description'])['TotalCost'].sum().groupby('Country', group_keys=False).nlargest(5)
        top_prods = pd.DataFrame(top_prods)
        top_prods.reset_index(inplace=True)

        fig = go.Figure()
        for country in top_prods['Country'].unique():
            temp = top_prods[top_prods['Country'] == country]
            fig.add_trace(go.Bar(x=temp['Description'], y=temp['TotalCost'], name=country))

        steps = []
        for i, country in enumerate(top_prods['Country'].unique()):
            step = dict(
                method="update",
                label=country,
                args=[{"visible": [country == trace.name for trace in fig.data]}],
            )
            steps.append(step)

        sliders = [dict(active=0, currentvalue={'prefix' : 'Country: '}, pad={"t": 50},
                    steps=steps)]

        fig.update_layout(
            sliders=sliders,
            title="Top Products by Country According to Revenue",
            xaxis_title="Product",
            yaxis_title="Revenue"
        )

        fig.show()
        self.savePlot(fig, 'topProductsCountry.html')
        return fig

    def topProductsCountryQtyPlt(self): #top products according to country plot
        top_prods = self.df1.groupby(['Country', 'Description'])['Quantity'].sum().groupby('Country', group_keys=False).nlargest(5)
        top_prods = pd.DataFrame(top_prods)
        top_prods.reset_index(inplace=True)

        fig = go.Figure()
        for country in top_prods['Country'].unique():
            temp = top_prods[top_prods['Country'] == country]
            fig.add_trace(go.Bar(x=temp['Description'], y=temp['Quantity'], name=country))

        steps = []
        for i, country in enumerate(top_prods['Country'].unique()):
            step = dict(
                method="update",
                label=country,
                args=[{"visible": [country == trace.name for trace in fig.data]}],
            )
            steps.append(step)

        sliders = [dict(active=0, currentvalue={'prefix' : 'Country: '}, pad={"t": 50},
                    steps=steps)]

        fig.update_layout(
            sliders=sliders,
            title="Top Products by Country According to Quantity Sold",
            xaxis_title="Product",
            yaxis_title="Quantity ordered"
        )

        fig.show()
        self.savePlot(fig, 'topProductsCountryQty.html')
        return fig

    def topProdsWorldWidePlt(self):
        series = self.df1['Description'].value_counts(ascending=False).nlargest(10)
        trace = go.Bar(x=series.index, y=series.values)
        fig = go.Figure([trace])
        fig.update_layout(xaxis_title='Product Description', yaxis_title='Qty Sold',
        title='Top 10 Products Sold Worldwide')
        fig.show()
        self.savePlot(fig, 'top10ProductsWorld.html')
        return fig

    def top5ProdsYearlyPlt(self):
        self.df1['InvoiceDate'] = pd.to_datetime(self.df1['InvoiceDate'])
        series = self.df1.groupby(self.df1['InvoiceDate'].dt.year).apply(lambda x: x.groupby('Description')['TotalCost'].sum().nlargest(5))
        top_prods = pd.DataFrame(series)
        top_prods.reset_index(inplace=True)

        fig = go.Figure()
        for year in top_prods['InvoiceDate'].unique():
            temp = top_prods[top_prods['InvoiceDate'] == year]
            fig.add_trace(go.Bar(x=temp['Description'], y=temp['TotalCost'], name=str(year)))
        
        steps = []
        for i, year in enumerate(top_prods['InvoiceDate'].unique()):
            step = dict(method='update', label=str(year), 
            args=[{'visible' : [str(year) == trace.name for trace in fig.data]}])
            steps.append(step)

        sliders = [dict(active=0, currentvalue={'prefix' : 'Year: '}, pad={"t": 50},
                    steps=steps)]

        fig.update_layout(
            sliders=sliders,
            title="Top Products by Country According to Total Cost",
            xaxis_title="Product",
            yaxis_title="Revenue")

        fig.show()
        self.savePlot(fig, 'top5ProductsWorldYearly.html')
        return fig
    
    def worldwideTopNProds(self, n=5):
        top_products = self.df1.groupby('Description')['Quantity'].sum().sort_values(ascending=False)[:n]
        traces = []
        for product in top_products.index:
            temp = self.df1[self.df1['Description'] == product]
            temp['InvoiceDate'] = pd.to_datetime(temp['InvoiceDate'])
            temp['Month'] = temp['InvoiceDate'].dt.month
            monthly_cost = temp.groupby('Month')['Quantity'].sum()
            trace = go.Scatter(x=monthly_cost.index, y=monthly_cost,
                            mode='lines+markers', name=product)
            traces.append(trace)

        fig = go.Figure(traces)
        fig.update_layout(title="Total Cost per Month for Top 5 Products", 
                        xaxis_title="Month", yaxis_title="Total Cost")
        fig.show()
        self.savePlot(fig, 'worldTopNProducts.html')
        return fig

    def topCustFavProdsPlot(self):
        top_customers = self.df1.groupby('CustomerID')['TotalCost'].sum().nlargest(5)
        orders = {}

        for customer in top_customers.index:
            temp = self.df1[self.df1['CustomerID'] == customer]
            count = Counter(temp['Description']).most_common(5)
            orders[customer] = dict(count)

        fig = go.Figure()

        for customer in orders.keys():
            products_names = orders[customer].keys()
            product_count = orders[customer].values()
            fig.add_trace(go.Bar(x=list(products_names), y=list(product_count), name=str(customer)))

        steps = []
        for i, id in enumerate(orders.keys()):
            step = dict(method='update', label=str(id), 
            args=[{'visible' : [str(id) == trace.name for trace in fig.data]}])
            steps.append(step)

        sliders = [dict(active=0, currentvalue={'prefix' : 'CustomerID: '}, pad={"t": 50},
                    steps=steps)]

        fig.update_layout(
            sliders=sliders,
            title="Top 5 Products by Customers with highest business",
            xaxis_title="Product",
            yaxis_title="Revenue")

        fig.show()
        self.savePlot(fig, 'FavProdsofTopCustomers.html')
        return fig

    def topSpendingCountry(self):
        otldf = self.df1.copy()
        #ret = self.df1[(self.df1['TotalCost'] < 0) & (~self.df1['Description'].isin(['Unsaleable, destroyed.', 'Damaged', 'damages']))]
        topCountries = self.df1.groupby('Country')['TotalCost'].sum().nlargest(5).index
        traces = []
        for country in topCountries:
            temp = self.df1[self.df1["Country"]==country]
            trace = go.Box(y=temp['TotalCost'], name=country)
            traces.append(trace)
        
        fig = go.Figure(traces)
        fig.show()
        self.savePlot(fig, 'top5SpendingCountry.html')
        return fig

    def mostDefectiveProducts(self): #loss incurred per product for a defect
        defective = ['Unsaleable, destroyed.', 'Damaged', 'damages', 'mouldy, unsaleable.', 'wet boxes', 'wet', 'wet rusty']
        defective_df = self.df1[self.df1['Description'].isin(defective)]
        return np.abs(defective_df.groupby('StockCode')['TotalCost'].sum().nsmallest(7))
        
    
    def lostProducts(self): #loss incurred per product for the product being lost
        products = pickle.load(open('products.pk', 'rb'))
        lostTags = [prod for prod in set(self.df1['Description']) if ('lost' in prod) or ('mixed' in prod) or ('?' in prod)]
        lostdf = self.df1[self.df1['Description'].isin(lostTags)]
        return np.abs(lostdf.groupby('StockCode')['TotalCost'].sum().nsmallest(10))
        

    def lossDuetoRetTopProds(self): #top products total sales and return
        sales, ret = self.df1[self.df1['Quantity'] > 0], self.df1[self.df1['Quantity'] < 0]
        sales = pd.DataFrame(sales.groupby('Description')['Quantity'].sum()).reset_index()
        ret = pd.DataFrame(ret.groupby('Description')['Quantity'].sum()).reset_index()
        merged = sales.merge(ret, left_on='Description', right_on='Description')
        merged.rename(columns={'Quantity_x': 'Actual Sales', 'Quantity_y': 'QtyReturned'}, inplace=True)
        merged.sort_values(by='Actual Sales', ascending=False, inplace=True)
        merged['NetSales'] = merged['Actual Sales'] + merged['QtyReturned']
        x = merged['Description'].tolist()[:10]
        y = merged['Actual Sales'].tolist()[:10]
        y1 = [abs(qty) for qty in merged['NetSales'].tolist()[:10]]
        trace1 = go.Bar(x=x, y=y, name='Actual Sales', marker_color='#1f77b4')
        trace2 = go.Bar(x=x, y=y1, name='Net Sales after Return', marker_color='#ff7f0e')
        fig = go.Figure([trace1, trace2])
        fig.update_layout(xaxis_title='Product name', yaxis_title='Total Qty Sold vs Qty Returned',
        title='Potential Sales of Top Products VS Net Sales after Return')
        fig.show()
        self.savePlot(fig, 'lossDuetoTopProdsReturn.html')
        return fig
    
    def savePlot(self, fig, filename):
        fig.write_html(filename+'.html')
    
plts = MakePlots('cleanDf.csv')
plts.topProductsPerCountryPlt()
plts.topProductsCountryQtyPlt()
plts.topProdsWorldWidePlt()
plts.top5ProdsYearlyPlt()
plts.worldwideTopNProds()
plts.topCustFavProdsPlot()
plts.lossDuetoRetTopProds()
print(plts.lostProducts())
print(plts.mostDefectiveProducts())