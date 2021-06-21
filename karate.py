#!/usr/bin/env python

import networkx as nx
import dimod
import matplotlib.pyplot as plt
from dwave.system import LeapHybridDQMSampler
import numpy as np
from random import random


G = nx.karate_club_graph()

dqm = dimod.DiscreteQuadraticModel()
n_clusters = 4
n_nodes = G.number_of_nodes()
beta = 3

for i in G.nodes:
    dqm.add_variable(n_clusters, label=i)

norm_factor = beta * (1 - (2 * n_nodes / n_clusters))
for i in G.nodes:
    linear_term = norm_factor + (0.5 * np.ones(n_clusters) * G.degree[i])
    dqm.set_linear(i, linear_term)

# Quadratic term for node pairs which do not have edges between them
for p0, p1 in nx.non_edges(G):
    dqm.set_quadratic(p0, p1, {(c, c): (2 * beta) for c in range(n_clusters)})

# Quadratic term for node pairs which have edges between them
for p0, p1 in G.edges:
    dqm.set_quadratic(p0, p1, {(c, c): ((2 * beta) - 1) for c in range(n_clusters)})

sampler = LeapHybridDQMSampler()
sampleset = sampler.sample_dqm(dqm, label='Example - Graph Partitioning DQM')

sample = sampleset.first.sample
energy = sampleset.first.energy

color_list = [(random(), random(), random()) for i in range(n_clusters)]
color_map = [color_list[sample[i]] for i in G.nodes]
nx.draw(G, node_color=color_map)
plt.savefig('graph_partition_result.png')
plt.close()
