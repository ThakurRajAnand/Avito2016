"""
USAGE: python create_item_network_id.py train|test
writes output to item_network_id_train|test.csv
"""
import pandas as pd
import csv
from collections import Counter
import sys
from itertools import chain

data_type = sys.argv[1]

input_filename = '../input/ItemPairs_{}.csv'.format(data_type)
output_filename = 'item_network_id_{}.csv'.format(data_type)

# First pass
items = {}
with open(input_filename, 'rb') as infile:
    reader = csv.reader(infile)
    reader.next()
    for count, line in enumerate(reader, 1):
        i1, i2 = map(int, line[:2])
        if i1 not in items:
            items[i1] = []
        if i2 not in items:
            items[i2] = []
        items[i1].append(i2)
        items[i2].append(i1)
        if count % 10000 == 0:
            sys.stderr.write('\r--- First pass {:,}'.format(count))
            sys.stderr.flush()
sys.stderr.write('\n')

# Create dataframe of itemID and network_id
output = []
network_id = 0
for this_key in sorted(items.keys()):
    if this_key not in items:
        continue
    prior_count = 0
    new_items = {this_key: items[this_key]}
    del items[this_key]
    while len(new_items) > prior_count:
        prior_count = len(new_items)
        for i in chain(*new_items.itervalues()):
            new_item = items.get(i, '')
            if new_item:
                new_items[i] = new_item
                del items[i]
    for i in new_items.keys():
        values = {'itemID': i, 'network_id': network_id}
        output.append(values)
    network_id += 1
    sys.stderr.write('\r--- {:,} Processed; {:,} Remaining        '.format(len(output), len(items)))
    sys.stderr.flush()
sys.stderr.write('\n')

df_network_id = pd.DataFrame(output)

# Merge to ItemsPair and output itemID_1, itemID_2, network_id to stdout
sys.stderr.write('--- Reading {} and merging\n'.format(input_filename))
pairs = pd.read_csv(input_filename)
pairs = pairs.merge(df_network_id, how='inner', left_on='itemID_1', right_on='itemID')
sys.stderr.write('--- Calculating network size and merging\n')
network_size = pairs.groupby('network_id').size().reset_index()
network_size.columns = ['network_id', 'network_size']
pairs = pairs.merge(network_size, how='inner', on='network_id')
sys.stderr.write('--- Writing {}\n'.format(output_filename))
pairs[['itemID_1', 'itemID_2', 'network_id', 'network_size']].to_csv(output_filename, index=False)
