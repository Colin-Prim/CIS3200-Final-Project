#  https://www.kaggle.com/code/mapologo/loading-wikipedia-math-essentials
import json

import numpy as np
import pandas as pd
import networkx as nx

#  loading data
with open("wikivital_mathematics.json") as f:
    data = json.load(f)

#  numpy array of edges
edges = np.array(data['edges'])
#  numpy array of all weights
weights = np.array(data['weights'])
#  numpy array of all weighted edges
weighted_edges = [edge + [weight] for edge, weight in zip(data['edges'], data['weights'])]

#  make a df of datetime instances for later use when finding daily visitation
time_periods = {}
for time_period in range(data['time_periods']):
    time_periods[time_period] = data[str(time_period)]
dates = pd.DataFrame.from_dict(time_periods, orient="index", columns=['year', 'month', 'day'])

#  find daily visitation of each page
#  so far this is a 1068 x 2 matrix, with each row containing a page index in col1 and an array of daily
#    visitation in col2
daily_visits = pd.DataFrame.from_dict(time_periods, orient="index", columns=['y'])
#  now convert the daily visits from a 1067 x 2 matrix into a 731 x 1068 matrix
#    each row represents a datetime instance
#    each column represents a page
#  such that daily_visits(r, c) represents the number of visitations of page c on day r
#  visit number are assumed to be exact and not scaled
daily_visits = pd.DataFrame(daily_visits.y.to_list())
daily_visits = daily_visits.set_index(pd.to_datetime(dates))

#  allows finding of position (pos) from page name
#  sample syntax: pages_pos.loc['Mathematics'].pos yields 0
pages_pos = pd.DataFrame.from_dict(data['node_ids'], orient='index')
pages_pos.columns = ["pos"]
pages_pos = pages_pos.sort_values("pos")

#  allows finding of page name from position (pos)
#  sample syntax: pos_pages.loc[0].page yields 'Mathematics'
pos_pages = pd.DataFrame(pages_pos.index, index=pages_pos.pos)
pos_pages.columns = ['page']

#  init graph from weighted_edges
G = nx.DiGraph()
G.add_weighted_edges_from(weighted_edges)

#  rank pages and convert to a numpy array
ranked = nx.pagerank(G)
ranked_array = np.array(list(ranked.items()))
#print(ranked_array)  # comment out to speed up program

pagerank_sorted = sorted(ranked.items(), key=lambda item:item[1], reverse=True)
sorted_array = np.array(list(pagerank_sorted))
#print(sorted_array[:10])

print("Default Parameters:")
n=0
while n < 10:
    index = sorted_array[n][0]
    print(pos_pages.loc[index].page)
    n += 1

total_visits = daily_visits.sum(axis=0)

new_weighted_edges = [(from_node, to_node, total_visits.get(to_node, np.nan)) for from_node, to_node in edges]

G_new = nx.DiGraph()
G_new.add_weighted_edges_from(new_weighted_edges)

new_ranked = nx.pagerank(G_new)
new_ranked_array = np.array(list(new_ranked.items()))

new_sorted = sorted(new_ranked.items(), key=lambda item:item[1], reverse=True)
new_array = np.array(list(new_sorted))

print("Default Parameters:")
n=0
while n < 10:
    index = new_array[n][0]
    print(pos_pages.loc[index].page)
    n += 1