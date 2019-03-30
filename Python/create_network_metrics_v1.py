from collections import Counter
import csv
import glob
import itertools
import multiprocessing
import os
import re
import sys
import time

import pandas as pd
import numpy as np
import networkx as nx

data_type = sys.argv[1] # train or test

# Read and sort
lines = []
with open('item_network_id_{:}.csv'.format(data_type), 'rb') as infile:
    reader = csv.DictReader(infile)
    for count, line in enumerate(reader, 1):
        lines.append(line)
        if count % 10000 == 0:
            print '\r--- Reading item pairs {:,}'.format(count),
            sys.stdout.flush()
lines.sort(key=lambda x: x['network_id'])
print

# Create list of grouped lines by network_id
grouped_lines = []
network_id_prior = lines[0]['network_id']
current_stack = []
for count, line in enumerate(lines, 1):
    if count % 10000 == 0:
        print '\r--- Grouping networks {:,}'.format(count),
        sys.stdout.flush()
    if line['network_id'] == network_id_prior:
        current_stack.append(line)
    else:
        grouped_lines.append(current_stack[:])
        network_id_prior = line['network_id']
        current_stack = [line]
grouped_lines.append(current_stack[:])
print

def get_metrics(df):
    """Returns pageranks, average_neighbor_degree, average_node_connectivity"""
    values = df[['itemID_1', 'itemID_2']].values
    G = nx.Graph()
    G.add_nodes_from(set(values.flatten()))
    G.add_edges_from(values)
    pageranks = nx.pagerank(G) # this is dict
    avg_neighbors = nx.average_neighbor_degree(G) # this is dict
    avg_node_connectivity = nx.average_node_connectivity(G) # This is a scalar
    return pageranks, avg_neighbors, avg_node_connectivity

def create_data(lines):
    df = pd.DataFrame(lines)

    pageranks, avg_neighbors, avg_node_connectivity = get_metrics(df)
    for k in pageranks:
        pageranks[k] = re.sub('0+$', '0', '{:.6f}'.format(pageranks[k]))
    for k in avg_neighbors:
        avg_neighbors[k] = re.sub('0+$', '0', '{:.6f}'.format(avg_neighbors[k]))
    df['pagerank_1'] = df['itemID_1'].map(pageranks)
    df['pagerank_2'] = df['itemID_2'].map(pageranks)
    df['avg_neighbor_degree_1'] = df['itemID_1'].map(avg_neighbors)
    df['avg_neighbor_degree_2'] = df['itemID_2'].map(avg_neighbors)
    df['avg_node_connectivity'] = re.sub('0+$', '0', '{:.6f}'.format(avg_node_connectivity))

    return df.T.to_dict().values()

#grouped_lines = grouped_lines[:1000]

num_tasks = len(grouped_lines)

p = multiprocessing.Pool()
results = p.imap(create_data, grouped_lines)
while (True):
    completed = results._index
    print "\r--- Completed {:,} out of {:,}".format(completed, num_tasks),
    sys.stdout.flush()
    time.sleep(1)
    if (completed == num_tasks): break
p.close()
p.join()
with open('features_network_v1_{:}.csv'.format(data_type), 'wb') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=[u'itemID_1', u'itemID_2', u'network_id', u'network_size', u'pagerank_1',
                                                   u'pagerank_2', u'avg_neighbor_degree_1', u'avg_neighbor_degree_2',
                                                   u'avg_node_connectivity'])
    writer.writeheader()
    for r in results:
        writer.writerows(r)
