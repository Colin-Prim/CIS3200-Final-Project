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

total_visits = daily_visits.sum(axis=0)
avg_total_visits = total_visits.mean()

#  allows finding of position (pos) from page name
#  sample syntax: pages_pos.loc['Mathematics'].pos yields 0
pages_pos = pd.DataFrame.from_dict(data['node_ids'], orient='index')
pages_pos.columns = ["pos"]
pages_pos = pages_pos.sort_values("pos")

#  allows finding of page name from position (pos)
#  sample syntax: pos_pages.loc[0].page yields 'Mathematics'
pos_pages = pd.DataFrame(pages_pos.index, index=pages_pos.pos)
pos_pages.columns = ['page']

'''Default Weighting (Num Recurrences)'''
#  init graph from weighted_edges
G = nx.DiGraph()
G.add_weighted_edges_from(weighted_edges)

#  rank pages and convert to a numpy array
ranked = nx.pagerank(G)

# sort the ranked pages in ascending order by pagerank and convert to array
pagerank_sorted = sorted(ranked.items(), key=lambda item: item[1], reverse=True)
sorted_array = np.array(list(pagerank_sorted))
# print(sorted_array) # comment out to speed up program


# display the top 10 pages based on default parameters and normal weights
print("Default Parameters with Normal Weights:")
n = 0
while n < 10:
    index = sorted_array[n][0]
    print(pos_pages.loc[index].page)
    n += 1

'''Weighting by Total Visitation of Incoming Node'''
new_weighted_edges = [(from_node, to_node, total_visits.get(to_node, np.nan)) for from_node, to_node in edges]

G_new = nx.DiGraph()
G_new.add_weighted_edges_from(new_weighted_edges)

new_ranked = nx.pagerank(G_new)
new_ranked_array = np.array(list(new_ranked.items()))

# sort the ranked pages from pagerank based on total visitation weights and convert to array
new_sorted = sorted(new_ranked.items(), key=lambda item: item[1], reverse=True)
new_array = np.array(list(new_sorted))

print("\n" + "Weighted by Total Visitation (Incoming):")
n = 0
while n < 10:
    index = new_array[n][0]
    print(pos_pages.loc[index].page)
    n += 1

# pagerank using visitation weights had 1 page in common with the top 10 most visited pages
# pagerank using normal weights had 0 pages in common with the top 10 most visited pages
# we will adjust the parameters in pagerank using visitation weights to see if we can get more accurate results

# changing alpha to 0.95 instead of the default (0.85)
variation_1 = nx.pagerank(G_new, alpha=0.95)
variation_1_array = np.array(list(variation_1.items()))

# sort ranked pages and convert to array
variation_1_sorted = sorted(variation_1.items(), key=lambda item: item[1], reverse=True)
variation_1_sorted_array = np.array(list(variation_1_sorted))

# display top 10 pages when changing alpha to 0.95
print("Alpha = 0.95")
n = 0
while n < 10:
    index = variation_1_sorted_array[n][0]
    print(pos_pages.loc[index].page)
    n += 1

# changing alpha to 0.75
variation_2 = nx.pagerank(G_new, alpha=0.75)
variation_2_array = np.array(list(variation_2.items()))

# sort ranked pages and convert to array
variation_2_sorted = sorted(variation_2.items(), key=lambda item: item[1], reverse=True)
variation_2_sorted_array = np.array(list(variation_2_sorted))

# display top 10 pages when changing alpha to 0.75
print("Alpha = 0.75:")
n = 0
while n < 10:
    index = variation_2_sorted_array[n][0]
    print(pos_pages.loc[index].page)
    n += 1

# changing error tolerance to 1e-02 instead of the default (1e-06)
variation_3 = nx.pagerank(G_new, tol=1e-02)
variation_3_array = np.array(list(variation_3.items()))

# sort ranked pages and convert to array
variation_3_sorted = sorted(variation_3.items(), key=lambda item: item[1], reverse=True)
variation_3_sorted_array = np.array(list(variation_3_sorted))

# display top 10 pages when changing error tolerance to 1e-02
print("Error Tolerance = .01:")
n = 0
while n < 10:
    index = variation_3_sorted_array[n][0]
    print(pos_pages.loc[index].page)
    n += 1

'''Weighting by Total Visitation of Outgoing Node'''
new_weighted_edges = [(from_node, to_node, total_visits.get(from_node, np.nan)) for from_node, to_node in edges]

G_new = nx.DiGraph()
G_new.add_weighted_edges_from(new_weighted_edges)

new_ranked = nx.pagerank(G_new)

new_sorted = sorted(new_ranked.items(), key=lambda item: item[1], reverse=True)
new_array = np.array(list(new_sorted))

print("\n" + "Weighted by Total Visitation (Outgoing):")
n = 0
while n < 10:
    index = new_array[n][0]
    print(pos_pages.loc[index].page)
    n += 1

'''Weighting by Avg Total Visitation of Incoming and Outgoing Nodes'''
new_weighted_edges = [(from_node, to_node, (total_visits.get(from_node, np.nan) + total_visits.get(to_node, np.nan))
                       / 2) for from_node, to_node in edges]

G_new = nx.DiGraph()
G_new.add_weighted_edges_from(new_weighted_edges)

new_ranked = nx.pagerank(G_new)

new_sorted = sorted(new_ranked.items(), key=lambda item: item[1], reverse=True)
new_array = np.array(list(new_sorted))

print("\n" + "Weighted by Avg Total Visitation (Incoming/Outgoing):")
n = 0
while n < 10:
    index = new_array[n][0]
    print(pos_pages.loc[index].page)
    n += 1

'''This method of formatting produces a graph on which the PageRank algorithm does not converge'''
# '''Weighting by Total Visitation of Incoming Node Minus Average Total Visitation'''
# new_weighted_edges = [(from_node, to_node, total_visits.get(to_node, np.nan) - avg_total_visits)
#                       for from_node, to_node in edges]
#
# G_new = nx.DiGraph()
# G_new.add_weighted_edges_from(new_weighted_edges)
#
# new_ranked = nx.pagerank(G_new)
#
# new_sorted = sorted(new_ranked.items(), key=lambda item: item[1], reverse=True)
# new_array = np.array(list(new_sorted))
#
# print("\n" + "Weighted by Total Visitation (Incoming) Minus Average Visitation:")
# n = 0
# while n < 10:
#     index = new_array[n][0]
#     print(pos_pages.loc[index].page)
#     n += 1


'''Weighting by Total Visitation of Outgoing Node Minus Average Total Visitation'''
new_weighted_edges = [(from_node, to_node, total_visits.get(from_node, np.nan) - avg_total_visits)
                      for from_node, to_node in edges]

G_new = nx.DiGraph()
G_new.add_weighted_edges_from(new_weighted_edges)

new_ranked = nx.pagerank(G_new)

new_sorted = sorted(new_ranked.items(), key=lambda item: item[1], reverse=True)
new_array = np.array(list(new_sorted))

print("\n" + "Weighted by Total Visitation (Outgoing) Minus Average Visitation:")
n = 0
while n < 10:
    index = new_array[n][0]
    print(pos_pages.loc[index].page)
    n += 1
