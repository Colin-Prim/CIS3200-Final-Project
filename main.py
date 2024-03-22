#  https://www.kaggle.com/code/mapologo/loading-wikipedia-math-essentials
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open("wikivital_mathematics.json") as f:
    data = json.load(f)

print(data.keys())
