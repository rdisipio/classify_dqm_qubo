#!/usr/bin/env python

import random

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

import dimod


stop_words = stopwords.words('english')
embed_dim = 32
topk = 5
n_docs_max = 20
n_topics = 5

tfIdfVectorizer = TfidfVectorizer(use_idf=True, 
                                  stop_words=stop_words,
                                  max_df=1.0, min_df=1,
                                  max_features=embed_dim)

dataset = fetch_20newsgroups().data
dataset = random.sample(dataset, n_docs_max)
print(f"Number of entries: {len(dataset)}")

tfIdf = tfIdfVectorizer.fit_transform(dataset)

similarities = cosine_similarity(tfIdf[0:1], tfIdf).flatten()
related_docs_indices = similarities.argsort()[:-topk:-1]
print(related_docs_indices)
print(similarities[related_docs_indices])

reduced_data = PCA(n_components=2).fit_transform(tfIdf.todense())
kmeans = KMeans(init="k-means++", n_clusters=n_topics, n_init=4, random_state=12345)
cluster_labels = kmeans.fit_predict(reduced_data)
score = silhouette_score(reduced_data, cluster_labels)
print(f'N clusters: {n_topics}, score = {score:.2f}')

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation="nearest",
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired, aspect="auto", origin="lower")

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=1)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3,
            color="w", zorder=10)
plt.title("K-means clustering on the digits dataset (PCA-reduced data)\n"
          "Centroids are marked with white cross")
plt.xlim(0.5*x_min, 0.5*x_max)
plt.ylim(0.5*y_min, 0.5*y_max)
plt.xticks(())
plt.yticks(())
plt.show()
plt.savefig('newsgroups_kmeans.png')
plt.close()

