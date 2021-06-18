#!/usr/bin/env python

import networkx as nx
import dimod
import matplotlib.pyplot as plt
from dwave.system import LeapHybridDQMSampler


G = nx.karate_club_graph()

dqm = dimod.DiscreteQuadraticModel()
num_partitions = 4
num_nodes = G.number_of_nodes()
lagrange = 3

for i in G.nodes:
    dqm.add_variable(num_partitions, label=i)

constraint_const = lagrange * (1 - (2 * num_nodes / num_partitions))

for i in G.nodes:
    linear_term = constraint_const + (0.5 * np.ones(num_partitions) * G.degree[i])
    dqm.set_linear(i, linear_term)

# Quadratic term for node pairs which do not have edges between them
for p0, p1 in nx.non_edges(G):
    dqm.set_quadratic(p0, p1, {(c, c): (2 * lagrange) for c in range(num_partitions)})

# Quadratic term for node pairs which have edges between them
for p0, p1 in G.edges:
    dqm.set_quadratic(p0, p1, {(c, c): ((2 * lagrange) - 1) for c in range(num_partitions)})

sampleset = LeapHybridDQMSampler().sample_dqm(dqm)

color_list = [(random(), random(), random()) for i in range(num_partitions)]
color_map = [color_list[sample[i]] for i in G.nodes]
nx.draw(G, node_color=color_map)
plt.savefig('graph_partition_result.png')
plt.close()
