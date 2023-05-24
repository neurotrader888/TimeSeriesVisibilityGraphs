import pandas as pd
import numpy as np
import scipy

def network_prediction(network, data, times=None):
    '''
    May 21, 2023. This code will be covered in a future video.
    Implementation of this paper:

    Zhan, Tianxiang & Xiao, Fuyuan. (2021). 
    A novel weighted approach for time series forecasting based on visibility graph. 

    https://arxiv.org/abs/2103.13870
    '''
    if times is None:
        times = np.arange(len(data))

    n = len(data)
    degrees = np.sum(network, axis=1)
    num_edges = np.sum(network) # Number of edges * 2 (network is symmetric)

    # Transition probability matrix
    p = network.copy()
    for x in range(n):
        p[x, :] /= degrees[x]

    # Forecast vector
    forecasts = np.zeros(n -1)
    v = data[n-1]
    for x in range(n-1): # Forecast slope, not next val
        forecasts[x] = (v - data[x]) / (times[n-1] - times[x])

    srw = np.zeros(n-1)
    lrw_last = None
    walk_x = np.identity(n)
    t = 1
    while True:
        for x in range(n):
            walk_x[x,:] = np.dot(p.T, walk_x[x,:])
       
        # Find similarity with last node (most recent value)
        lrw = np.zeros(n-1) # -1 because not including last
        y = n - 1
        for x in range(n-1):
            lrw[x] =  (degrees[x] / num_edges) * walk_x[x, y]
            lrw[x] += (degrees[y] / num_edges) * walk_x[y, x]

        srw += lrw 
        if (lrw == lrw_last).all():
            #print(t)
            break
        lrw_last = lrw
        t += 1
        if t > 1000:
            break

    forecast_weights = srw / np.sum(srw)
    forecast = np.dot(forecast_weights, forecasts)

    return forecast
