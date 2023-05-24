import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches

def ts_to_vg(data: np.array, times: np.array = None, horizontal: bool = False):
    # Convert timeseries to visibility graph with DC algorithm

    if times is None:
        times = np.arange(len(data))

    network_matrix = np.zeros((len(data), len(data)))

    # DC visablity graph func
    def dc_vg(x, t, left, right, network):
        if left >= right:
            return
        k = np.argmax(x[left:right+1]) + left # Max node in left-right
        #print(left, right, k)
        for i in range(left, right+1):
            if i == k:
                continue

            visible = True
            for j in range(min(i+1, k+1), max(i, k)):
                # Visiblity check, EQ 1 from paper 
                if horizontal:
                    if x[j] >= x[i]:
                        visible = False
                        break
                else:
                    if x[j] >= x[i] + (x[k] - x[i]) * ((t[j] - t[i]) / (t[k] - t[i])):
                        visible = False
                        break

            if visible:
                network[k, i] = 1.0
                network[i, k] = 1.0
        
        dc_vg(x, t, left, k - 1, network) 
        dc_vg(x, t, k + 1, right, network) 

    dc_vg(data, times, 0, len(data) - 1, network_matrix)
    return network_matrix

def plot_ts_visibility(network: np.array, data: np.array, times: np.array = None, horizontal: bool = False):
    if times is None:
        times = np.arange(len(data))

    plt.style.use('dark_background') 
    fig, axs = plt.subplots(2, 1, sharex=True)
    # Plot connections and series
    for i in range(len(data)):
        for j in range(i, len(data)):
            if network[i, j] == 1.0:
                if horizontal:
                    axs[0].plot([times[i], times[j]], [data[i], data[i]], color='red', alpha=0.8)
                    axs[0].plot([times[i], times[j]], [data[j], data[j]], color='red', alpha=0.8)
                else:
                    axs[0].plot([times[i], times[j]], [data[i], data[j]], color='red', alpha=0.8)
    axs[0].plot(times, data)
    #axs[0].bar(times, data, width=0.1)
    axs[0].get_xaxis().set_ticks(list(times))

    # Plot graph
    for i in range(len(data)):
        axs[1].plot(times[i], 0, marker='o', color='orange')

    for i in range(len(data)):
        for j in range(i, len(data)):
            if network[i, j] == 1.0:
                Path = mpath.Path
                mid_time = (times[j] + times[i]) / 2.
                diff = abs(times[j] - times[i])
                pp1 = mpatches.PathPatch(Path([(times[i], 0), (mid_time, diff), (times[j], 0)],[Path.MOVETO, Path.CURVE3, Path.CURVE3]), fc="none", transform=axs[1].transData, alpha=0.5)
                axs[1].add_patch(pp1)
    axs[1].get_yaxis().set_ticks([])
    axs[1].get_xaxis().set_ticks(list(times))
    plt.show()

if __name__ == '__main__':

    # Example data from video
    dat = np.array([20, 40, 48, 70, 40, 60, 40, 100, 40, 80])
    #dat = np.array([.71, .53, .56, .29, .30, .77, .01, .76, .81, .71, .05, .41, .86, .79, .37, .96, .87, .06, .95, .36])

    # Cosine wave with 4 cycles (period = 12)
    #dat = np.cos( 2 * np.pi * (1/12) * np.arange(48) )

    # Daily bitcoin data, 
    #bitcoin_data = pd.read_csv('BTCUSDT86400.csv')
    # Decemeber 2022
    #dat = bitcoin_data['close'].iloc[-31:].to_numpy()
    
    # Network is the adjacency matrix
    network = ts_to_vg(dat)
  
    # Print adjacency matrix
    network = network.astype(int)
    index = list(range(len(dat)))
    index = [str(x) for x in index] 
    print("    " + " ".join(index))
    print("    " + "-" * (len(dat) * 2 - 1))
    for i in range(len(dat)):
        row = f"{i} | {str(network[:, i])[1:-1]}"
        print(row)

    plot_ts_visibility(network, dat, horizontal=False)



