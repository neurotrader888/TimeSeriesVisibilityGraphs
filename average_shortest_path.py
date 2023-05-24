import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from ts2vg import HorizontalVG, NaturalVG
from ts_to_vg import plot_ts_visibility
from network_indicators import shortest_path_length

data = pd.read_csv('BTCUSDT3600.csv')
data['date'] = data['date'].astype('datetime64[s]')
data = data.set_index('date')

lookback = 12
close_arr = data['close'].to_numpy()
pos, neg = shortest_path_length(close_arr, lookback)
data['pos'] = pos
data['neg'] = neg

data = data.dropna().reset_index()

# Plot visibility graph with max and min avearge_shortest_path
max_idx = data['pos'].idxmax()
min_idx = data['pos'].idxmin()

max_dat = data.iloc[max_idx - lookback +1: max_idx+1]['close'].to_numpy()
min_dat = data.iloc[min_idx - lookback +1: min_idx+1]['close'].to_numpy()

g = NaturalVG()
g.build(max_dat)
plot_ts_visibility(g.adjacency_matrix(), max_dat)


g = NaturalVG()
g.build(min_dat)
plot_ts_visibility(g.adjacency_matrix(), min_dat)

np.log(data['close']).plot()
plt.twinx()
data['neg'].plot(color='red', label='neg', alpha=0.8)
data['pos'].plot(color='green',label='pos', alpha=0.8)
plt.legend()
plt.show()


