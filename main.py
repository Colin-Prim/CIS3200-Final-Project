#  https://www.kaggle.com/code/mapologo/loading-wikipedia-math-essentials
import json

import numpy as np
import pandas as pd
import networkx as nx

#  loading data
#with open("wikivital_mathematics.json") as f:
#    data = json.load(f)
'''
    Make load data function so I can call it in my ipynb???
    Instead of loading the data like we did above ^^^
'''
def get_data():
    '''Returns a dict of data from data file'''
    with open("wikivital_mathematics.json") as f:
        data = json.load(f)
    return data
#data = get_data()

if __name__ == "__main__":
    # load data
    data = get_data()
    
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
    print(ranked_array)  # comment out to speed up program