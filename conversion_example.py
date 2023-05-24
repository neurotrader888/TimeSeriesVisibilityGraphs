import pandas as pd
import numpy as np
import scipy

# pip install ts2vg
from ts2vg import HorizontalVG, NaturalVG
from ts_to_vg import plot_ts_visibility # Plotting function for visibility graph

dat = np.array([20, 40, 48, 70, 40, 60, 40, 100, 40, 80])
ng = NaturalVG()
ng.build(dat)

adj_matrix = ng.adjacency_matrix()
print(adj_matrix)

plot_ts_visibility(adj_matrix, dat)


#hg = HorizontalVG()
#hg.build(dat)
#plot_ts_visibility(hg.adjacency_matrix(), dat, horizontal=True)


