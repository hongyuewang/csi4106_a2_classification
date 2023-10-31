import pandas as pd
import itertools
import numpy as np

import utils

url = "https://raw.githubusercontent.com/AvaneeshM/WineDataset/main/WineQT.csv"

dataset = pd.read_csv(url)

print(dataset.columns)
print(dataset.head(10))

dataset = dataset.dropna()
string_to_list = utils.string_to_list

