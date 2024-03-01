# Retail-Transactions-Analysis

This repository contains:

1. Product Recommender in Python (named as ‘ProductRecommender.py’ in the Zip), that uses Cosine Similarity to find similar customers for a given CustomerID and recommend products based on the purchase history of the similar customers, you just need to pass the CustomerID for whom you want product recommendation.

2. Cluster Analysis to group similar products (products with similar names), and finding out Top Products and Top Markets (Countries) for each cluster in terms of Revenue contribution, refer to python file named ‘ProductClusters.py'.

3. Potential Revenue VS Net Revenue due to returned/undelivered products, and calculated the loss incurred due to the damaged/lost products, refer to ‘PreprocessAndPlot.py'.

I’ve ensured that the Titles of each chart are self explanatory (there might be cases where charts containing sliders appear messy when opened, just move the slider to the next step and the chart should appear normal thereafter), and added comments to the code wherever there were complex calculations. The folder contains couple of pickle(.pk) files just to save the models.
