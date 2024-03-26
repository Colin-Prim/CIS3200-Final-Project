#  https://www.kaggle.com/code/mapologo/loading-wikipedia-math-essentials
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open("wikivital_mathematics.json") as f:
    data = json.load(f)

edges = np.array(data['edges'])
weights = np.array(data['weights'])
weighted_edges = [edge + [weight] for edge, weight in zip(data['edges'], data['weights'])]

time_periods = {}
for time_period in range(data['time_periods']):
    time_periods[time_period] = data[str(time_period)]
dates = pd.DataFrame.from_dict(time_periods, orient="index", columns=['year', 'month', 'day'])

daily_visits = pd.DataFrame.from_dict(time_periods, orient="index", columns=['y'])

pages_pos = pd.DataFrame.from_dict(data['node_ids'], orient='index')
pages_pos.columns = ["pos"]
pages_pos = pages_pos.sort_values("pos")

pos_pages = pd.DataFrame(pages_pos.index, index=pages_pos.pos)
pos_pages.columns = ['page']

daily_visits = pd.DataFrame(daily_visits.y.to_list())
daily_visits = daily_visits.set_index(pd.to_datetime(dates))
