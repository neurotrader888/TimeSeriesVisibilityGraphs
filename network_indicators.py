import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from ts2vg import HorizontalVG, NaturalVG


def shortest_path_length(close: np.array, lookback: int):

    avg_short_dist_p = np.zeros(len(close))
    avg_short_dist_n = np.zeros(len(close))

    avg_short_dist_p[:] = np.nan
    avg_short_dist_n[:] = np.nan

    for i in range(lookback, len(close)):
        dat = close[i - lookback + 1: i+1]

        pos = NaturalVG()
        pos.build(dat)

        neg = NaturalVG()
        neg.build(-dat)

        neg = neg.as_networkx()
        pos = pos.as_networkx()
    
        # you could replace shortest_path_length with other networkx metrics..
        avg_short_dist_p[i] = nx.average_shortest_path_length(pos)
        avg_short_dist_n[i] = nx.average_shortest_path_length(neg)
        
        # Another possibility...
        #nx.degree_assortativity_coefficient(pos)
        #nx.degree_assortativity_coefficient(neg)
        # All kinds of stuff here
        # https://networkx.org/documentation/stable/reference/algorithms/index.html 
    return avg_short_dist_p, avg_short_dist_n


if __name__ == '__main__':
    data = pd.read_csv('BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    
    # Next log return


    data['r'] = np.log(data['close']).diff().shift(-1)
    # Compute shortest average path length for last 12 vals
    pos, neg = shortest_path_length(data['close'].to_numpy(), 12)
    data['pos'] = pos
    data['neg'] = neg

    # Compute signals
    data['long_sig'] = 0
    data['short_sig'] = 0
    data.loc[data['pos'] > data['neg'], 'long_sig'] = 1
    data.loc[data['pos'] < data['neg'], 'short_sig'] = -1
    data['combined_sig'] = data['long_sig'] + data['short_sig']

    # Compute returns 
    data['long_ret'] = data['long_sig'] * data['r']
    data['short_ret'] = data['short_sig'] * data['r']
    data['comb_ret'] = data['combined_sig'] * data['r']

    # Compute profit factor
    long_pf = data[data['long_ret'] > 0]['long_ret'].sum() / data[data['long_ret'] < 0]['long_ret'].abs().sum()
    short_pf = data[data['short_ret'] > 0]['short_ret'].sum() / data[data['short_ret'] < 0]['short_ret'].abs().sum()
    combined_pf = data[data['comb_ret'] > 0]['comb_ret'].sum() / data[data['comb_ret'] < 0]['comb_ret'].abs().sum()
    
    print("Long PF", long_pf)
    print("Short PF", short_pf)
    print("Combined PF", combined_pf)
    
    # Plot cumulative log return
    plt.style.use('dark_background')
    data['long_ret'].cumsum().plot(label='Long')
    data['short_ret'].cumsum().plot(label='Short')
    data['comb_ret'].cumsum().plot(label='Combined')
    plt.legend()
    plt.show()
    



    '''
    #heatmap stuff
    import seaborn as sns
    heatmap_df = pd.DataFrame()
   
    for lb in range(6, 25):
        print(lb)
        # Compute shortest average path length for last 12 vals
        pos, neg = shortest_path_length(data['close'].to_numpy(), lb)
        data['pos'] = pos
        data['neg'] = neg

        # Compute signals
        data['long_sig'] = 0
        data['short_sig'] = 0
        data.loc[data['pos'] > data['neg'], 'long_sig'] = 1
        data.loc[data['pos'] < data['neg'], 'short_sig'] = -1
        data['combined_sig'] = data['long_sig'] + data['short_sig']

        # Compute returns 
        data['long_ret'] = data['long_sig'] * data['r']
        data['short_ret'] = data['short_sig'] * data['r']
        data['comb_ret'] = data['combined_sig'] * data['r']

        # Compute profit factor
        long_pf = data[data['long_ret'] > 0]['long_ret'].sum() / data[data['long_ret'] < 0]['long_ret'].abs().sum()
        short_pf = data[data['short_ret'] > 0]['short_ret'].sum() / data[data['short_ret'] < 0]['short_ret'].abs().sum()
        combined_pf = data[data['comb_ret'] > 0]['comb_ret'].sum() / data[data['comb_ret'] < 0]['comb_ret'].abs().sum()

        heatmap_df.loc[lb, 'Long'] = long_pf
        heatmap_df.loc[lb, 'Short'] = short_pf
        heatmap_df.loc[lb, 'Combined'] = combined_pf
    
    plt.style.use('dark_background')
    sns.heatmap(heatmap_df.T, annot=True)
    plt.xlabel("Visibility Graph Lookback")
    '''



    
