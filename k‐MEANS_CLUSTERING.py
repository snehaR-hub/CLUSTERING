'K-Means clustering is an unsupervised learning algorithm that is used to group data into K clusters based on feature similarity. Each data point belongs to the cluster whose centroid is the nearest.

'''Steps for the demonstration:
Install required libraries: We'll use scikit-learn for K-Means and matplotlib for visualizations.
Generate a synthetic dataset: We'll generate some random data for clustering.
Apply K-Means Clustering: Use the K-Means algorithm to fit the data and create clusters.
Visualize the Clusters: Plot the data points and the centroids of the clusters.'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Step 1: Generate a synthetic dataset with 2 features (2D points)
# We generate 300 data points with 4 centers (clusters)
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Step 2: Apply K-Means Clustering
# We want to divide the data into 4 clusters
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# Step 3: Get the cluster centers and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Step 4: Visualize the Clusters
plt.figure(figsize=(8,6))

# Scatter plot of the data points with different colors for each cluster
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)

# Plot the centroids (the center of each cluster) with a red star marker
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='*', label='Centroids')

# Labels and title
plt.title('K-Means Clustering with 4 Clusters', fontsize=14)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
